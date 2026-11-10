"""
Trains the Lab Agent on FSH, LH, AMH (+ FSH/LH ratio).

Four classifiers (Random Forest, XGBoost, SVM, Decision Tree) are compared on
accuracy / precision / recall / F1 / specificity / AUC; the best one (by mean
cross-validated F1) is saved to LAB_MODEL_PATH and used by LabAgent.

Run: python -m mapcos.training.train_lab_agent
"""

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from mapcos.config import (
    TABULAR_XLSX_PATH, TABULAR_SHEET_NAME, TARGET_COLUMN,
    LAB_RAW_COLUMNS, LAB_MODEL_PATH
)
from mapcos.training.tabular_compare import compare_and_select

OUT_DIR = "lab_comparison"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    fsh = pd.to_numeric(df[LAB_RAW_COLUMNS["fsh"]], errors="coerce")
    lh = pd.to_numeric(df[LAB_RAW_COLUMNS["lh"]], errors="coerce")
    amh = pd.to_numeric(df[LAB_RAW_COLUMNS["amh"]], errors="coerce")
    fsh_lh_ratio = fsh / lh.replace(0, 1e-6)
    return pd.DataFrame({"fsh": fsh, "lh": lh, "fsh_lh_ratio": fsh_lh_ratio, "amh": amh})


def main(xlsx_path=TABULAR_XLSX_PATH, sheet_name=TABULAR_SHEET_NAME,
         model_path=LAB_MODEL_PATH, out_dir=OUT_DIR):
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)

    X = build_features(df)
    y = df[TARGET_COLUMN]

    valid_rows = X.notna().all(axis=1) & y.notna()
    dropped = (~valid_rows).sum()
    if dropped > 0:
        print(f"Dropping {dropped} rows with non-numeric lab values")
    X, y = X[valid_rows], y[valid_rows].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train rows: {len(X_train)}, test rows: {len(X_test)}, "
          f"PCOS share in train: {y_train.mean():.2%}")

    best_name, best_model, cv_df, test_df = compare_and_select(
        X_train, y_train, X_test, y_test, out_dir=out_dir, tag="lab")

    joblib.dump(best_model, model_path)
    print(f"Best model ({best_name}) saved to {model_path}")
    return best_name, best_model


if __name__ == "__main__":
    main()
