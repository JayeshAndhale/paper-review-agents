"""Precision/recall/F1 over a labeled verdict set.

Kept general over a `labels` list rather than hardcoded to the three verdict
strings, so the confusion-matrix machinery isn't verification-specific.
The checkpointed orchestration that actually calls the live verifier lives
in scripts/run_evaluation.py, not here -- a TPD wall can kill a run
partway through (see that script's docstring), so that loop needs to be
resumable, and duplicating a second, non-resumable version of it here would
just be a second way to lose progress.
"""

from collections import Counter
from dataclasses import dataclass


@dataclass
class ClassScore:
    label: str
    precision: float
    recall: float
    f1: float
    support: int  # number of ground-truth items with this true label


def confusion_matrix(y_true: list[str], y_pred: list[str]) -> Counter:
    """Counter keyed (true_label, pred_label) -> count."""
    return Counter(zip(y_true, y_pred))


def precision_recall_f1(y_true: list[str], y_pred: list[str], labels: list[str]) -> list[ClassScore]:
    matrix = confusion_matrix(y_true, y_pred)
    scores = []
    for label in labels:
        tp = matrix[(label, label)]
        fp = sum(count for (t, p), count in matrix.items() if p == label and t != label)
        fn = sum(count for (t, p), count in matrix.items() if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        support = sum(count for (t, _), count in matrix.items() if t == label)
        scores.append(ClassScore(label, round(precision, 3), round(recall, 3), round(f1, 3), support))
    return scores
