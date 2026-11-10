"""
Reusable helper: compares Random Forest, XGBoost, SVM and Decision Tree on a
tabular binary-classification problem and picks the best one.

How the comparison is done (so the choice is not made on the test set):
  1. 5-fold stratified cross-validation on the TRAIN split -> mean +/- std of
     accuracy, precision, recall, F1, specificity and AUC for every model.
  2. The winner is the model with the highest mean CV F1 (ties: mean CV AUC).
  3. All four models are then refit on the full train split and scored once on
     the untouched TEST split, so the table you report is unbiased.
Positive class = 1 (PCOS).
"""

import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, make_scorer,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

METRICS = ["accuracy", "precision", "recall", "f1", "specificity", "auc"]

SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "specificity": make_scorer(recall_score, pos_label=0, zero_division=0),
    "auc": "roc_auc",
}


def candidate_models(y_train, seed=42):
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05, eval_metric="logloss",
            scale_pos_weight=neg / max(pos, 1), random_state=seed),
        "SVM": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed)),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, class_weight="balanced", random_state=seed),
    }


def holdout_metrics(model, X, y, threshold=0.5):
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "f1": f1_score(y, preds, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "auc": roc_auc_score(y, probs) if len(set(y)) > 1 else float("nan"),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def _plot(test_df, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [m for m in METRICS if m != "auc"]
    ax = test_df[metrics].plot(kind="bar", figsize=(9, 4.5), ylim=(0, 1.05), rot=0)
    ax.set_title(title)
    ax.set_ylabel("score")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def compare_and_select(X_train, y_train, X_test, y_test, out_dir, tag="model", seed=42):
    os.makedirs(out_dir, exist_ok=True)
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    n_splits = max(2, min(5, int(y_train.value_counts().min())))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    models_ = candidate_models(y_train, seed)
    cv_mean, cv_std, test_rows, fitted = {}, {}, {}, {}

    for name, model in models_.items():
        print(f"Evaluating {name} ...")
        res = cross_validate(clone(model), X_train, y_train, cv=cv, scoring=SCORING)
        cv_mean[name] = {m: float(np.mean(res[f"test_{m}"])) for m in METRICS}
        cv_std[name] = {m: float(np.std(res[f"test_{m}"])) for m in METRICS}

        model.fit(X_train, y_train)
        fitted[name] = model
        test_rows[name] = holdout_metrics(model, X_test, y_test)

    cv_mean_df = pd.DataFrame(cv_mean).T[METRICS]
    cv_std_df = pd.DataFrame(cv_std).T[METRICS]
    test_df = pd.DataFrame(test_rows).T

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    pretty = cv_mean_df.round(3).astype(str) + " ± " + cv_std_df.round(3).astype(str)
    print(f"\n{n_splits}-fold CROSS-VALIDATION on the training split (mean ± std) — used to pick the winner")
    print(pretty)
    print("\nHOLD-OUT TEST metrics (positive class = PCOS)")
    print(test_df[METRICS].round(3))
    print("\nTEST confusion counts")
    print(test_df[["tn", "fp", "fn", "tp"]].astype(int))

    ranked = cv_mean_df.sort_values(["f1", "auc"], ascending=False)
    best = ranked.index[0]
    print(f"\n>>> Best model (highest mean CV F1, then AUC): {best}")

    cv_mean_df.to_csv(os.path.join(out_dir, f"{tag}_cv_mean.csv"))
    cv_std_df.to_csv(os.path.join(out_dir, f"{tag}_cv_std.csv"))
    test_df.to_csv(os.path.join(out_dir, f"{tag}_test_metrics.csv"))
    _plot(test_df, os.path.join(out_dir, f"{tag}_test_comparison.png"),
          f"{tag}: model comparison on the hold-out test set")
    with open(os.path.join(out_dir, f"{tag}_best.json"), "w") as f:
        json.dump({"best_model": best, "selected_by": "mean cross-validated F1",
                   "cv_metrics": cv_mean_df.loc[best].to_dict(),
                   "test_metrics": {k: float(v) for k, v in test_df.loc[best].items()}}, f, indent=2)

    return best, fitted[best], cv_mean_df, test_df
