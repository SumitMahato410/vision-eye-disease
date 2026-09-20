"""Evaluate a trained model on the held-out test split.

    python -m src.evaluate --task internal

Writes to reports/:
    <task>_metrics.json        accuracy / macro + weighted P, R, F1 / per class
    <task>_confusion_matrix.png
    <task>_confusion_matrix.csv
    <task>_training_curves.png  (if a history file exists)

These are the numbers and figures to paste into the Results chapter of the
report, so everything is written to disk rather than only printed.
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")  # headless: no display needed on Colab or a server
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from tensorflow import keras

from . import config as cfg
from .data import load_classes, load_split


def plot_confusion(cm: np.ndarray, classes: list[str], title: str, out_png):
    """Row-normalised confusion matrix: each row shows where a true class went."""
    with np.errstate(invalid="ignore", divide="ignore"):
        norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    norm = np.nan_to_num(norm)

    fig, ax = plt.subplots(figsize=(1.6 * len(classes) + 2, 1.4 * len(classes) + 2))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(
                j, i, f"{norm[i, j]:.2f}\n({cm[i, j]})",
                ha="center", va="center", fontsize=9,
                color="white" if norm[i, j] > 0.55 else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_curves(history: dict, title: str, out_png):
    keys = [k for k in ("accuracy", "loss", "auc") if k in history]
    if not keys:
        return
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4), squeeze=False)
    for ax, k in zip(axes[0], keys):
        ax.plot(history[k], label=f"train {k}")
        if f"val_{k}" in history:
            ax.plot(history[f"val_{k}"], label=f"val {k}")
        ax.set_xlabel("epoch")
        ax.set_ylabel(k)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def evaluate(task_key: str, split: str = "test") -> dict:
    task = cfg.get_task(task_key)
    if not task.model_path.exists():
        raise FileNotFoundError(
            f"No trained model at {task.model_path}. Run: python -m src.train --task {task_key}"
        )

    classes = load_classes(task)
    model = keras.models.load_model(task.model_path)
    ds = load_split(task, split, classes, shuffle=False)

    probs = model.predict(ds, verbose=1)
    y_true = np.concatenate([np.argmax(y.numpy(), axis=1) for _, y in ds])
    y_pred = np.argmax(probs, axis=1)

    report = classification_report(
        y_true, y_pred, target_names=classes, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))

    # One-vs-rest macro AUC, the metric that actually reflects performance on
    # the rare classes in an imbalanced dataset.
    try:
        auc = float(roc_auc_score(
            np.eye(len(classes))[y_true], probs, average="macro", multi_class="ovr"
        ))
    except ValueError:
        auc = None  # a class missing from the test split

    metrics = {
        "task": task.key,
        "split": split,
        "n_images": int(len(y_true)),
        "classes": classes,
        "accuracy": float(report["accuracy"]),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "macro_auc_ovr": auc,
        "per_class": {
            c: {
                "precision": float(report[c]["precision"]),
                "recall": float(report[c]["recall"]),
                "f1": float(report[c]["f1-score"]),
                "support": int(report[c]["support"]),
            }
            for c in classes
        },
        "confusion_matrix": cm.tolist(),
    }

    cfg.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (cfg.REPORT_DIR / f"{task.key}_metrics.json").write_text(json.dumps(metrics, indent=2))

    np.savetxt(
        cfg.REPORT_DIR / f"{task.key}_confusion_matrix.csv",
        cm, fmt="%d", delimiter=",",
        header=",".join(classes), comments="",
    )
    plot_confusion(
        cm, classes, f"{task.key} — confusion matrix ({split})",
        cfg.REPORT_DIR / f"{task.key}_confusion_matrix.png",
    )

    if task.history_path.exists():
        hist = json.loads(task.history_path.read_text()).get("history", {})
        plot_curves(hist, f"{task.key} — training curves",
                    cfg.REPORT_DIR / f"{task.key}_training_curves.png")

    # Console summary
    print("\n" + classification_report(y_true, y_pred, target_names=classes, zero_division=0))
    print(f"accuracy        {metrics['accuracy']:.4f}")
    print(f"macro F1        {metrics['macro_f1']:.4f}")
    if auc is not None:
        print(f"macro AUC (ovr) {auc:.4f}")
    print(f"\nArtifacts written to {cfg.REPORT_DIR}")
    return metrics


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Evaluate a trained VISION model.")
    p.add_argument("--task", required=True, choices=sorted(cfg.TASKS))
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    a = p.parse_args(argv)
    evaluate(a.task, a.split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
