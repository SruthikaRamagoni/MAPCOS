"""Evaluation helpers: accuracy, precision, recall, F1, specificity, ROC-AUC, confusion matrix, calibration."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


def calibration_bins(y_true, y_prob, n_bins: int = 10):
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    edges = np.linspace(0, 1, n_bins + 1)
    rows, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (y_prob >= lo) & ((y_prob < hi) if hi < 1 else (y_prob <= hi))
        if m.sum() == 0:
            continue
        conf, acc = float(y_prob[m].mean()), float(y_true[m].mean())
        rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()), "mean_confidence": conf, "observed_rate": acc})
        ece += m.mean() * abs(conf - acc)
    return rows, float(ece)


def full_report(y_true, y_prob, threshold: float = 0.5, n_bins: int = 10) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    bins, ece = calibration_bins(y_true, y_prob, n_bins)
    return {
        "n": int(len(y_true)), "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else float("nan"),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "brier": float(brier_score_loss(y_true, y_prob)), "ece": ece, "calibration_bins": bins,
    }
