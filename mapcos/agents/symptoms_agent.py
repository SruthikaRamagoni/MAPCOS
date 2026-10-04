"""
Trains the Symptoms Agent on cycle regularity, hirsutism, BMI and related
symptom flags.

Four classifiers (Random Forest, XGBoost, SVM, Decision Tree) are compared on
accuracy / precision / recall / F1 / specificity / AUC; the best one (by mean
cross-validated F1) is saved to SYMPTOMS_MODEL_PATH and used by SymptomsAgent.

Run: python -m mapcos.training.train_symptoms_agent
"""

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from mapcos.config import (
    TABULAR_XLSX_PATH, TABULAR_SHEET_NAME, TARGET_COLUMN,
    SYMPTOMS_RAW_COLUMNS, SYMPTOMS_MODEL_PATH
)
from mapcos.training.tabular_compare import compare_and_select

OUT_DIR = "symptoms_comparison"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    c = SYMPTOMS_RAW_COLUMNS
    cycle_irregular = (df[c["cycle_code"]] != 2).astype(int)
    feats = pd.DataFrame({
        "bmi": df[c["bmi"]],
        "cycle_irregular": cycle_irregular,
        "hair_growth": df[c["hair_growth"]],
        "weight_gain": df[c["weight_gain"]],
        "skin_darkening": df[c["skin_darkening"]],
        "hair_loss": df[c["hair_loss"]],
        "pimples": df[c["pimples"]],
    })
    return feats.apply(pd.to_numeric, errors="coerce")


def main(xlsx_path=TABULAR_XLSX_PATH, sheet_name=TABULAR_SHEET_NAME,
         model_path=SYMPTOMS_MODEL_PATH, out_dir=OUT_DIR):
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)

    X = build_features(df)
    y = df[TARGET_COLUMN]

    valid_rows = X.notna().all(axis=1) & y.notna()
    dropped = (~valid_rows).sum()
    if dropped > 0:
        print(f"Dropping {dropped} rows with missing / non-numeric symptom values")
    X, y = X[valid_rows], y[valid_rows].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train rows: {len(X_train)}, test rows: {len(X_test)}, "
          f"PCOS share in train: {y_train.mean():.2%}")

    best_name, best_model, cv_df, test_df = compare_and_select(
        X_train, y_train, X_test, y_test, out_dir=out_dir, tag="symptoms")

    joblib.dump(best_model, model_path)
    print(f"Best model ({best_name}) saved to {model_path}")
    return best_name, best_model


if __name__ == "__main__":
    main()
