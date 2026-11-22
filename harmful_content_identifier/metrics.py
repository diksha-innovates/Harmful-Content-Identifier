"""Evaluation metrics beyond plain accuracy/loss.

The original notebook only ever printed running loss and accuracy, which
hides how the model does on individual (especially minority) classes. This
module adds per-class precision/recall/F1, a confusion matrix, and a
human-readable classification report.
"""
import json
from pathlib import Path
from typing import List, Sequence

from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int], class_names: List[str]) -> dict:
    labels = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )

    per_class = {
        class_names[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in labels
    }

    return {
        "per_class": per_class,
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def classification_report_text(y_true: Sequence[int], y_pred: Sequence[int], class_names: List[str]) -> str:
    return classification_report(
        y_true, y_pred, labels=list(range(len(class_names))), target_names=class_names, zero_division=0
    )


def save_metrics(metrics: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def plot_confusion_matrix(confusion: Sequence[Sequence[int]], class_names: List[str], path: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cm = np.array(confusion)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
