"""Binary-classification metrics shared by federated clients."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def binary_classification_metrics(
    labels: Iterable[float], predictions: Iterable[float]
) -> dict[str, float | int]:
    """Return robust binary metrics using a fixed 0/1 label convention."""
    y_true = np.asarray(list(labels), dtype=int).reshape(-1)
    y_pred = np.asarray(list(predictions), dtype=int).reshape(-1)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "num_samples": int(y_true.size),
    }


def loader_class_distribution(loader) -> dict[str, int]:
    """Read binary class counts from this project's TensorDataset-backed loader."""
    labels = loader.dataset.tensors[1].detach().cpu().numpy().reshape(-1).astype(int)
    positives = int(labels.sum())
    return {
        "samples": int(labels.size),
        "positive_samples": positives,
        "negative_samples": int(labels.size - positives),
    }
