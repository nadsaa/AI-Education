# MathDial Teacher Intent Classification

This repository contains coursework and experiment artifacts for pedagogical intent classification on the MathDial tutoring-dialog dataset. The project compares lightweight encoder baselines in notebooks with a Qwen3-1.7B LoRA fine-tuning pipeline implemented as standalone Python scripts.

The core questions are:

- Can a model classify the intent of the current teacher utterance?
- Can a model predict the teacher's next intent from dialogue history alone?
- How do results change between a fine-grained 11-label setting and a coarse 4-label setting?

## Tasks

Two input variants are used throughout the project:

- `V1`: prior dialogue history + current teacher utterance -> predict the current teacher intent
- `V2`: prior dialogue history only -> predict the next teacher intent

Two label spaces are used:

- Fine-grained (`Predicted_Intents`, 11 labels): `Revealing Strategy`, `Revealing Answer`, `Guiding Student Focus`, `Seek Strategy`, `Asking for Explanation`, `Seeking Self Correction`, `Seeking World Knowledge`, `Greeting/Fairwell`, `Recall Relevant Information`, `Perturbing the Question`, `General inquiry`
- Coarse (`Intent`, 4 labels): `focus`, `generic`, `probing`, `telling`

## Repository Contents

This is primarily a research-code repository rather than a packaged Python library.

- `scripts/train_qwen3_lora_intent.py`: main training and evaluation script for Qwen3 LoRA experiments
- `scripts/plot_qwen3_lora_results.py`: generates training curves, metric comparisons, confusion matrices, and per-class F1 plots from saved run artifacts
- `dialogs_train_annotated.tsv`, `dialogs_val_annotated.tsv`, `dialogs_test_annotated.tsv`: MathDial-derived annotated splits used by all experiments
- `project_distilbert_v4.ipynb`: DistilBERT baseline notebook
- `project_tinybert_mathdial-2.ipynb`: TinyBERT baseline notebook


## Data

The TSV files contain quoted multi-line fields, so use a proper TSV parser such as `pandas.read_csv(..., sep="\t", on_bad_lines="skip")`, which is what the training script does.

Parsed split sizes in the current repository:

| Split | Dialog Rows | Conversations | Teacher Rows |
| --- | ---: | ---: | ---: |
| Train | 8,154 | 500 | 5,174 |
| Validation | 160 | 11 | 100 |
| Test | 152 | 10 | 99 |

Effective training examples are built from teacher turns only:

- Fine-grained runs use `Predicted_Intents` and produce `5,174 / 100 / 99` train/val/test examples
- Coarse runs use `Intent` and produce `5,173 / 100 / 99` train/val/test examples after filtering out rows whose label is not one of the four supported coarse classes

## Qwen3 LoRA Pipeline

The Qwen pipeline is implemented in `scripts/train_qwen3_lora_intent.py`.

Main behavior:

- Loads the annotated train/validation/test TSV files
- Builds teacher-turn samples for `V1`, `V2`, or both
- Fine-tunes `Qwen/Qwen3-1.7B` for sequence classification with LoRA
- Automatically targets `q_proj`, `k_proj`, `v_proj`, and `o_proj` unless overridden
- Supports three training regimes for the fine-grained setup:
  - Standard training
  - Oversampling (`--balance-train oversample`)
  - Class-weighted loss (`--balance-train class_weighted`)
- Saves the best adapter, checkpoints, metrics, predictions, label counts, confusion matrices, and a per-run config JSON

Default training hyperparameters:

| Setting | Value |
| --- | --- |
| Model | `Qwen/Qwen3-1.7B` |
| Epochs | `5` |
| Learning rate | `2e-4` |
| Train batch size | `2` |
| Eval batch size | `4` |
| Gradient accumulation | `8` |
| Max sequence length | `512` |
| LoRA rank | `16` |
| LoRA alpha | `32` |
| LoRA dropout | `0.1` |
| Seed | `44` |

## Environment Setup

There is no pinned `requirements.txt` in the repository. The SLURM jobs install the runtime dependencies directly in the target environment.

Minimal environment:

```bash
python -m pip install --upgrade "transformers<5,>=4.56.0" peft accelerate torch \
  scikit-learn pandas numpy matplotlib sacrebleu
```

Notes:

- First-time execution needs access to download `Qwen/Qwen3-1.7B` from Hugging Face
- The training script uses `trust_remote_code=True` for the Qwen model/tokenizer
- A GPU is strongly recommended; the cluster jobs request `1` GPU and `80G` host memory

## Running Experiments

### Local training

Standard 11-label training for both variants:

```bash
python scripts/train_qwen3_lora_intent.py \
  --train-file dialogs_train_annotated.tsv \
  --val-file dialogs_val_annotated.tsv \
  --test-file dialogs_test_annotated.tsv \
  --model-name Qwen/Qwen3-1.7B \
  --variant both \
  --output-dir outputs/qwen3_1p7b_lora_intent_seed44 \
  --epochs 5 \
  --learning-rate 2e-4 \
  --train-batch-size 2 \
  --eval-batch-size 4 \
  --grad-accumulation-steps 8 \
  --max-length 512 \
  --seed 44
```

Oversampled fine-grained training:

```bash
python scripts/train_qwen3_lora_intent.py \
  --variant both \
  --balance-train oversample \
  --output-dir outputs/qwen3_1p7b_lora_intent_balanced_seed44
```

Class-weighted fine-grained training:

```bash
python scripts/train_qwen3_lora_intent.py \
  --variant both \
  --balance-train class_weighted \
  --output-dir outputs/qwen3_1p7b_lora_intent_class_weighted_seed44
```

Coarse 4-label training:

```bash
python scripts/train_qwen3_lora_intent.py \
  --variant both \
  --label-space coarse \
  --label-column Intent \
  --output-dir outputs/qwen3_1p7b_lora_intent_coarse_seed44
```


## Plotting Results

To generate figures for a saved run:

```bash
python scripts/plot_qwen3_lora_results.py \
  --run-dir outputs/qwen3_1p7b_lora_intent_seed44
```

This creates a `figures/` directory inside the selected run folder. The current script is most useful for the fine-grained 11-label runs, which match its hard-coded label order.

## Output Layout

A completed Qwen run looks like this:

```text
outputs/qwen3_1p7b_lora_intent_seed44/
├── summary_metrics.json
├── figures/
│   ├── 01_train_loss_vs_step.png
│   ├── 02_validation_curves.png
│   ├── 03_metric_comparison.png
│   ├── 04_confusion_matrices.png
│   └── 05_per_class_f1.png
├── v1/
│   ├── best_model/
│   ├── checkpoints/
│   ├── classification_report.json
│   ├── classification_report.txt
│   ├── run_config.json
│   ├── test_confusion_matrix.csv
│   ├── test_metrics.json
│   ├── test_predictions.csv
│   ├── training_history.json
│   └── val_metrics.json
└── v2/
    └── ...
```

Key artifacts:

- `summary_metrics.json`: top-level summary for `v1` and `v2`
- `run_config.json`: exact settings and final metrics for one variant
- `test_predictions.csv`: per-example predictions
- `classification_report.*`: detailed class-level evaluation
- `best_model/`: saved LoRA adapter and tokenizer files

## Current Results Snapshot

### Fine-grained 11-label Qwen3 LoRA runs

| Training | Variant | Val Weighted F1 | Test Accuracy | Test Weighted F1 | Test Macro F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Standard | V1 | 0.6826 | 0.6162 | 0.6217 | 0.5473 |
| Standard | V2 | 0.3984 | 0.1616 | 0.1482 | 0.0996 |
| Class-weighted | V1 | 0.6521 | 0.6263 | 0.6250 | 0.5600 |
| Class-weighted | V2 | 0.3771 | 0.1717 | 0.1514 | 0.1104 |
| Oversampled | V1 | 0.6585 | **0.6364** | **0.6274** | **0.6225** |
| Oversampled | V2 | 0.3880 | 0.2323 | 0.1900 | 0.1267 |

### Coarse 4-label Qwen3 LoRA runs

| Variant | Val Weighted F1 | Test Accuracy | Test Weighted F1 | Test Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| V1 | 0.6839 | **0.7071** | **0.7077** | 0.6012 |
| V2 | **0.7052** | 0.5253 | 0.5420 | 0.5078 |

High-level takeaways reflected in the saved report tables:

- `V1` is consistently stronger than `V2`, especially in the 11-label setting
- The best 11-label test results come from Qwen3 LoRA with oversampling in `V1`
- The 4-label setup is substantially easier than the 11-label setup
- DistilBERT remains a competitive baseline, especially for coarse `V2`, but the strongest overall `V1` results in this repo come from Qwen3 LoRA

## Practical Notes

- The training script filters to teacher turns with labels in the active label set; unlabeled or malformed rows are skipped
- `V2` uses prior dialogue context only; when no context exists, the script inserts `[NO CONTEXT]`
- The plotting script expects the saved JSON/CSV artifact structure produced by `scripts/train_qwen3_lora_intent.py`
- This repository currently focuses on experimentation and reporting rather than packaging or deployment
