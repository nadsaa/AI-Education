#!/usr/bin/env python
"""
LoRA fine-tuning for Qwen3-1.7B intent classification on MathDial teacher turns.

This script mirrors the DistilBERT notebook data construction:
- V1: prior context + current teacher utterance
- V2: prior context only
"""

from __future__ import annotations

import argparse
import gc
import json
import inspect
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from peft import LoraConfig, TaskType, get_peft_model

try:
    import sacrebleu
except Exception:
    sacrebleu = None


INTENT_ORDER = [
    "Revealing Strategy",
    "Revealing Answer",
    "Guiding Student Focus",
    "Seek Strategy",
    "Asking for Explanation",
    "Seeking Self Correction",
    "Seeking World Knowledge",
    "Greeting/Fairwell",
    "Recall Relevant Information",
    "Perturbing the Question",
    "General inquiry",
]

LABEL2ID = {label: i for i, label in enumerate(INTENT_ORDER)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen/Qwen3-1.7B with LoRA for intent classification.")
    parser.add_argument("--train-file", type=str, default="dialogs_train_annotated.tsv")
    parser.add_argument("--val-file", type=str, default="dialogs_val_annotated.tsv")
    parser.add_argument("--test-file", type=str, default="dialogs_test_annotated.tsv")
    parser.add_argument("--output-dir", type=str, default="outputs/qwen3_1p7b_lora_intent")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--variant", type=str, default="both", choices=["v1", "v2", "both"])

    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--grad-accumulation-steps", type=int, default=8)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=44)

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument(
        "--target-modules",
        nargs="*",
        default=None,
        help="Override LoRA target modules; defaults to q/k/v/o projections when present.",
    )

    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument(
        "--balance-train",
        type=str,
        default="none",
        choices=["none", "oversample", "class_weighted"],
        help="Train split balancing strategy (applied only to train samples).",
    )
    parser.add_argument(
        "--balance-target-count",
        type=int,
        default=None,
        help="Target count per class for oversampling. Default: max class count in train split.",
    )
    return parser.parse_args()


def load_split(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", on_bad_lines="skip")


def build_dialog_samples(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    """
    Build independent teacher-turn samples from prior dialog context.
    """
    assert variant in {"v1", "v2"}
    samples: List[Dict[str, object]] = []

    for conv_no, conv_df in df.groupby("Conversation_No"):
        conv_df = conv_df.sort_values("Utterance_Index")
        turns = conv_df[["Speaker", "Text", "Predicted_Intents"]].values.tolist()

        for turn_idx, (speaker, text, pred_intent) in enumerate(turns):
            if speaker != "Teacher" or pd.isna(pred_intent) or pred_intent not in LABEL2ID:
                continue

            prior_parts: List[str] = []
            for prev_speaker, prev_text, prev_intent in turns[:turn_idx]:
                if prev_speaker == "Teacher" and pd.notna(prev_intent) and prev_intent in LABEL2ID:
                    prior_parts.append(f"Teacher [{prev_intent}]: {str(prev_text)}")
                elif prev_speaker == "Student":
                    prior_parts.append(f"Student: {str(prev_text)}")

            current_utterance = f"Teacher: {str(text)}"
            if variant == "v1":
                text_in = " [SEP] ".join(prior_parts + [current_utterance]) if prior_parts else current_utterance
            else:
                text_in = " [SEP] ".join(prior_parts) if prior_parts else "[NO CONTEXT]"

            samples.append(
                {
                    "text": text_in,
                    "label": pred_intent,
                    "label_id": LABEL2ID[pred_intent],
                    "conv": int(conv_no),
                    "turn": int(turn_idx),
                }
            )

    return pd.DataFrame(samples)


class IntentDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer: AutoTokenizer, max_length: int):
        self.frame = frame.reset_index(drop=True)
        self.encodings = tokenizer(
            self.frame["text"].astype(str).tolist(),
            truncation=True,
            max_length=max_length,
        )
        self.labels = self.frame["label_id"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def maybe_truncate(frame: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    if limit is None or limit >= len(frame):
        return frame
    return frame.iloc[:limit].copy()


def label_counts(frame: pd.DataFrame) -> Dict[str, int]:
    counts = frame["label"].value_counts().to_dict()
    return {label: int(counts.get(label, 0)) for label in INTENT_ORDER}


def maybe_balance_train_samples(
    train_samples: pd.DataFrame,
    method: str,
    seed: int,
    target_count: int | None,
) -> tuple[pd.DataFrame, Dict[str, int], Dict[str, int]]:
    before = label_counts(train_samples)
    if method in {"none", "class_weighted"}:
        return train_samples, before, before

    if method != "oversample":
        raise ValueError(f"Unsupported balance method: {method}")

    non_zero_counts = [c for c in before.values() if c > 0]
    if not non_zero_counts:
        return train_samples, before, before
    per_class_target = target_count if target_count is not None else max(non_zero_counts)
    if per_class_target <= 0:
        raise ValueError("balance_target_count must be positive.")

    rng = np.random.default_rng(seed)
    parts = []
    for label in INTENT_ORDER:
        subset = train_samples[train_samples["label"] == label]
        if subset.empty:
            continue
        idx = subset.index.to_numpy()
        sampled_idx = rng.choice(idx, size=per_class_target, replace=len(idx) < per_class_target)
        parts.append(train_samples.loc[sampled_idx])

    balanced = pd.concat(parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    after = label_counts(balanced)
    return balanced, before, after


def compute_inverse_freq_class_weights(counts_by_label: Dict[str, int]) -> Dict[str, float]:
    counts = np.array([counts_by_label.get(label, 0) for label in INTENT_ORDER], dtype=np.float64)
    nonzero = counts > 0
    if not np.any(nonzero):
        return {label: 0.0 for label in INTENT_ORDER}

    total = float(counts[nonzero].sum())
    num_nonzero = int(nonzero.sum())
    weights = np.zeros_like(counts, dtype=np.float64)
    weights[nonzero] = total / (num_nonzero * counts[nonzero])

    # Normalize to keep average non-zero weight at 1.0 for stable LR behavior.
    mean_nonzero = float(weights[nonzero].mean())
    if mean_nonzero > 0:
        weights[nonzero] = weights[nonzero] / mean_nonzero

    return {label: float(weights[i]) for i, label in enumerate(INTENT_ORDER)}


class WeightedLossTrainer(Trainer):
    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):  # noqa: ARG002
        if self.class_weights is None:
            return super().compute_loss(model, inputs, return_outputs=return_outputs)

        labels = inputs["labels"]
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**model_inputs)
        logits = outputs.get("logits") if isinstance(outputs, dict) else outputs.logits

        weight = self.class_weights.to(device=logits.device, dtype=logits.dtype)
        loss_fct = torch.nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def pick_target_modules(model: torch.nn.Module, user_choice: List[str] | None) -> List[str]:
    if user_choice:
        return user_choice

    default_candidates = ["q_proj", "k_proj", "v_proj", "o_proj"]
    available_leaf_names = {name.split(".")[-1] for name, _ in model.named_modules()}
    selected = [name for name in default_candidates if name in available_leaf_names]
    if not selected:
        raise ValueError(
            "Could not infer LoRA target modules. Pass --target-modules explicitly for this architecture."
        )
    return selected


def compute_metrics(eval_pred) -> Dict[str, float]:
    if hasattr(eval_pred, "predictions"):
        logits = eval_pred.predictions
        labels = eval_pred.label_ids
    else:
        logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "weighted_f1": f1_score(labels, preds, average="weighted"),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def write_json(path: Path, data: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=True)


def evaluate_and_save(
    trainer: Trainer,
    test_dataset: IntentDataset,
    test_samples: pd.DataFrame,
    variant_output_dir: Path,
) -> Dict[str, float]:
    prediction = trainer.predict(test_dataset)
    logits = prediction.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    labels = prediction.label_ids.astype(int)
    preds = np.argmax(logits, axis=-1).astype(int)

    test_accuracy = accuracy_score(labels, preds)
    test_weighted_f1 = f1_score(labels, preds, average="weighted")
    test_macro_f1 = f1_score(labels, preds, average="macro")
    test_bleu = None
    if sacrebleu is not None:
        pred_texts = [ID2LABEL[int(x)] for x in preds]
        ref_texts = [ID2LABEL[int(x)] for x in labels]
        test_bleu = float(sacrebleu.corpus_bleu(pred_texts, [ref_texts]).score)

    eval_metrics = {
        "test_accuracy": float(test_accuracy),
        "test_weighted_f1": float(test_weighted_f1),
        "test_macro_f1": float(test_macro_f1),
        "test_bleu": test_bleu,
    }

    fixed_labels = list(range(len(INTENT_ORDER)))
    report_text = classification_report(
        labels,
        preds,
        labels=fixed_labels,
        target_names=INTENT_ORDER,
        digits=4,
        zero_division=0,
    )
    report_json = classification_report(
        labels,
        preds,
        labels=fixed_labels,
        target_names=INTENT_ORDER,
        digits=4,
        zero_division=0,
        output_dict=True,
    )

    cm = confusion_matrix(labels, preds, labels=list(range(len(INTENT_ORDER))))
    cm_df = pd.DataFrame(cm, index=INTENT_ORDER, columns=INTENT_ORDER)
    cm_df.to_csv(variant_output_dir / "test_confusion_matrix.csv")

    pred_df = test_samples[["conv", "turn", "label"]].copy().reset_index(drop=True)
    pred_df["true_label"] = [ID2LABEL[int(x)] for x in labels]
    pred_df["pred_label"] = [ID2LABEL[int(x)] for x in preds]
    pred_df["correct"] = pred_df["true_label"] == pred_df["pred_label"]
    pred_df.to_csv(variant_output_dir / "test_predictions.csv", index=False)

    with (variant_output_dir / "classification_report.txt").open("w", encoding="utf-8") as f:
        f.write(report_text)
        f.write("\n")

    write_json(variant_output_dir / "classification_report.json", report_json)
    write_json(variant_output_dir / "test_metrics.json", eval_metrics)

    print("=" * 70)
    print(f"{variant_output_dir.name.upper()} test metrics")
    print("=" * 70)
    print(f"Accuracy    : {test_accuracy:.4f}")
    print(f"Weighted F1 : {test_weighted_f1:.4f}")
    print(f"Macro F1    : {test_macro_f1:.4f}")
    if test_bleu is None:
        print("BLEU        : n/a (install sacrebleu to enable)")
    else:
        print(f"BLEU        : {test_bleu:.4f}")
    print("-" * 70)
    print(report_text)
    return eval_metrics


def run_variant(
    args: argparse.Namespace,
    variant: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict[str, float]:
    print(f"\nPreparing samples for variant={variant}")
    raw_train_samples = build_dialog_samples(train_df, variant)
    train_samples, train_counts_before, train_counts_after = maybe_balance_train_samples(
        raw_train_samples,
        method=args.balance_train,
        seed=args.seed,
        target_count=args.balance_target_count,
    )
    train_samples = maybe_truncate(train_samples, args.max_train_samples)
    val_samples = maybe_truncate(build_dialog_samples(val_df, variant), args.max_val_samples)
    test_samples = maybe_truncate(build_dialog_samples(test_df, variant), args.max_test_samples)
    class_weights = None
    class_weights_dict = None
    if args.balance_train == "class_weighted":
        class_weights_dict = compute_inverse_freq_class_weights(train_counts_before)
        class_weights = torch.tensor(
            [class_weights_dict[label] for label in INTENT_ORDER],
            dtype=torch.float32,
        )

    print(
        f"{variant}: train={len(train_samples)} val={len(val_samples)} test={len(test_samples)} "
        f"labels={len(INTENT_ORDER)}"
    )
    print(f"{variant}: balance_train={args.balance_train}")
    if args.balance_train == "oversample":
        print(f"{variant}: train label counts before balancing = {train_counts_before}")
        print(f"{variant}: train label counts after balancing  = {train_counts_after}")
    elif args.balance_train == "class_weighted":
        print(f"{variant}: class weights = {class_weights_dict}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = IntentDataset(train_samples, tokenizer, args.max_length)
    val_dataset = IntentDataset(val_samples, tokenizer, args.max_length)
    test_dataset = IntentDataset(test_samples, tokenizer, args.max_length)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(INTENT_ORDER),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    target_modules = pick_target_modules(model, args.target_modules)
    modules_to_save = []
    for name, _ in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in {"score", "classifier"}:
            modules_to_save.append(leaf)
    modules_to_save = sorted(set(modules_to_save)) or None

    print(f"LoRA target modules for {variant}: {target_modules}")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
        modules_to_save=modules_to_save,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    variant_output_dir = Path(args.output_dir) / variant
    variant_output_dir.mkdir(parents=True, exist_ok=True)
    write_json(variant_output_dir / "train_label_counts_before.json", train_counts_before)
    write_json(variant_output_dir / "train_label_counts_after.json", train_counts_after)
    if class_weights_dict is not None:
        write_json(variant_output_dir / "class_weights.json", class_weights_dict)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8 if torch.cuda.is_available() else None)

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16 = torch.cuda.is_available() and not bf16

    training_kwargs = {
        "output_dir": str(variant_output_dir / "checkpoints"),
        "overwrite_output_dir": True,
        "num_train_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_weighted_f1",
        "greater_is_better": True,
        "save_total_limit": 2,
        "dataloader_num_workers": args.num_workers,
        "remove_unused_columns": False,
        "report_to": "none",
        "bf16": bf16,
        "fp16": fp16,
        "seed": args.seed,
    }
    if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters:
        training_kwargs["eval_strategy"] = "epoch"
    else:
        training_kwargs["evaluation_strategy"] = "epoch"

    training_args = TrainingArguments(**training_kwargs)

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    trainer.train()
    trainer.save_model(str(variant_output_dir / "best_model"))
    tokenizer.save_pretrained(str(variant_output_dir / "best_model"))

    history_path = variant_output_dir / "training_history.json"
    write_json(history_path, {"log_history": trainer.state.log_history})

    val_metrics = trainer.evaluate(val_dataset)
    write_json(variant_output_dir / "val_metrics.json", val_metrics)

    test_metrics = evaluate_and_save(trainer, test_dataset, test_samples, variant_output_dir)
    write_json(
        variant_output_dir / "run_config.json",
        {
            "variant": variant,
            "model_name": args.model_name,
            "balance_train": args.balance_train,
            "balance_target_count": args.balance_target_count,
            "class_weights": class_weights_dict,
            "train_samples_original": len(raw_train_samples),
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "test_samples": len(test_samples),
            "max_length": args.max_length,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "grad_accumulation_steps": args.grad_accumulation_steps,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "target_modules": target_modules,
            "seed": args.seed,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        },
    )

    del trainer
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "variant": variant,
        "val_weighted_f1": float(val_metrics.get("eval_weighted_f1", -1.0)),
        "val_macro_f1": float(val_metrics.get("eval_macro_f1", -1.0)),
        "test_accuracy": test_metrics["test_accuracy"],
        "test_weighted_f1": test_metrics["test_weighted_f1"],
        "test_macro_f1": test_metrics["test_macro_f1"],
        "test_bleu": test_metrics["test_bleu"],
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    train_df = load_split(args.train_file)
    val_df = load_split(args.val_file)
    test_df = load_split(args.test_file)

    variants: Iterable[str]
    if args.variant == "both":
        variants = ["v1", "v2"]
    else:
        variants = [args.variant]

    all_results = []
    for variant in variants:
        all_results.append(run_variant(args, variant, train_df, val_df, test_df))

    summary = {item["variant"]: item for item in all_results}
    write_json(Path(args.output_dir) / "summary_metrics.json", summary)

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    for row in all_results:
        print(
            f"{row['variant']}: "
                f"val_wf1={row['val_weighted_f1']:.4f} | "
                f"test_acc={row['test_accuracy']:.4f} | "
                f"test_wf1={row['test_weighted_f1']:.4f} | "
                f"test_mf1={row['test_macro_f1']:.4f} | "
                f"test_bleu={row['test_bleu'] if row['test_bleu'] is not None else 'n/a'}"
        )


if __name__ == "__main__":
    main()
