#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create visualizations for Qwen3 LoRA intent runs.")
    parser.add_argument(
        "--run-dir",
        type=str,
        default="outputs/qwen3_1p7b_lora_intent_seed44",
        help="Directory containing v1/, v2/, and summary_metrics.json",
    )
    parser.add_argument(
        "--fig-dir",
        type=str,
        default=None,
        help="Output figure directory (default: <run-dir>/figures)",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_log_history(run_dir: Path, variant: str) -> pd.DataFrame:
    hist = load_json(run_dir / variant / "training_history.json")["log_history"]
    return pd.DataFrame(hist)


def extract_eval_by_epoch(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["epoch", "eval_loss", "eval_accuracy", "eval_weighted_f1", "eval_macro_f1"]
    existing = [c for c in cols if c in df.columns]
    out = df[df["eval_loss"].notna()][existing].copy()
    return out.sort_values("epoch")


def plot_train_loss(fig_dir: Path, v1_df: pd.DataFrame, v2_df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 5))
    for name, df, color in [("V1", v1_df, "#1f77b4"), ("V2", v2_df, "#ff7f0e")]:
        train_rows = df[df["loss"].notna()]
        plt.plot(train_rows["step"], train_rows["loss"], label=f"{name} train loss", linewidth=2, color=color)
    plt.xlabel("Step")
    plt.ylabel("Train Loss")
    plt.title("Training Loss vs Step")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(fig_dir / "01_train_loss_vs_step.png", dpi=180)
    plt.close()


def plot_eval_curves(fig_dir: Path, v1_eval: pd.DataFrame, v2_eval: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    metrics = [
        ("eval_loss", "Validation Loss"),
        ("eval_weighted_f1", "Validation Weighted F1"),
        ("eval_macro_f1", "Validation Macro F1"),
    ]
    for ax, (m, title) in zip(axes, metrics):
        ax.plot(v1_eval["epoch"], v1_eval[m], marker="o", linewidth=2, label="V1", color="#1f77b4")
        ax.plot(v2_eval["epoch"], v2_eval[m], marker="s", linewidth=2, label="V2", color="#ff7f0e")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
        if "f1" in m:
            ax.set_ylim(0, 1)
    axes[0].set_ylabel("Value")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "02_validation_curves.png", dpi=180)
    plt.close()


def plot_summary_bars(fig_dir: Path, summary: dict) -> None:
    metrics = ["test_accuracy", "test_weighted_f1", "test_macro_f1", "val_weighted_f1", "val_macro_f1"]
    metric_names = ["Test Acc", "Test wF1", "Test mF1", "Val wF1", "Val mF1"]

    x = np.arange(len(metrics))
    width = 0.36
    v1 = [summary["v1"][m] for m in metrics]
    v2 = [summary["v2"][m] for m in metrics]

    plt.figure(figsize=(10.5, 5))
    plt.bar(x - width / 2, v1, width=width, label="V1", color="#1f77b4")
    plt.bar(x + width / 2, v2, width=width, label="V2", color="#ff7f0e")
    for i, val in enumerate(v1):
        plt.text(i - width / 2, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    for i, val in enumerate(v2):
        plt.text(i + width / 2, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    plt.xticks(x, metric_names)
    plt.ylim(0, 1.0)
    plt.title("V1 vs V2 Metrics")
    plt.ylabel("Score")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "03_metric_comparison.png", dpi=180)
    plt.close()


def _heatmap(ax, cm_df: pd.DataFrame, title: str) -> None:
    data = cm_df.values
    im = ax.imshow(data, cmap="Blues")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(cm_df.columns)))
    ax.set_yticks(np.arange(len(cm_df.index)))
    ax.set_xticklabels(cm_df.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(cm_df.index, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    thresh = data.max() * 0.5 if data.size else 0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = int(data[i, j])
            ax.text(j, i, str(v), ha="center", va="center", color="white" if v > thresh else "black", fontsize=6)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_confusion_matrices(fig_dir: Path, run_dir: Path) -> None:
    v1_cm = pd.read_csv(run_dir / "v1" / "test_confusion_matrix.csv", index_col=0)
    v2_cm = pd.read_csv(run_dir / "v2" / "test_confusion_matrix.csv", index_col=0)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    _heatmap(axes[0], v1_cm, "V1 Test Confusion Matrix")
    _heatmap(axes[1], v2_cm, "V2 Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(fig_dir / "04_confusion_matrices.png", dpi=180)
    plt.close()


def plot_per_class_f1(fig_dir: Path, run_dir: Path) -> None:
    v1_report = load_json(run_dir / "v1" / "classification_report.json")
    v2_report = load_json(run_dir / "v2" / "classification_report.json")

    v1 = [v1_report.get(label, {}).get("f1-score", 0.0) for label in INTENT_ORDER]
    v2 = [v2_report.get(label, {}).get("f1-score", 0.0) for label in INTENT_ORDER]

    x = np.arange(len(INTENT_ORDER))
    width = 0.38
    plt.figure(figsize=(15, 5))
    plt.bar(x - width / 2, v1, width=width, label="V1", color="#1f77b4")
    plt.bar(x + width / 2, v2, width=width, label="V2", color="#ff7f0e")
    plt.xticks(x, INTENT_ORDER, rotation=35, ha="right", fontsize=8)
    plt.ylabel("F1-score")
    plt.ylim(0, 1)
    plt.title("Per-class Test F1: V1 vs V2")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "05_per_class_f1.png", dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    fig_dir = Path(args.fig_dir) if args.fig_dir else run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    summary = load_json(run_dir / "summary_metrics.json")
    v1_df = load_log_history(run_dir, "v1")
    v2_df = load_log_history(run_dir, "v2")
    v1_eval = extract_eval_by_epoch(v1_df)
    v2_eval = extract_eval_by_epoch(v2_df)

    plot_train_loss(fig_dir, v1_df, v2_df)
    plot_eval_curves(fig_dir, v1_eval, v2_eval)
    plot_summary_bars(fig_dir, summary)
    plot_confusion_matrices(fig_dir, run_dir)
    plot_per_class_f1(fig_dir, run_dir)

    print(f"Saved figures to: {fig_dir}")
    for path in sorted(fig_dir.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
