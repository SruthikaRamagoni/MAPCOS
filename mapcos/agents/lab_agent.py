"""
Lab Agent
-----------
FSH, LH, AMH using the best classifier chosen by training/train_lab_agent.py
(Random Forest, XGBoost, SVM or Decision Tree). Any scikit-learn style model
with predict_proba works.
"""

import joblib
import pandas as pd


class LabAgent:
    def __init__(self, model_path: str = None, model=None):
        if model is not None:
            self.model = model
        elif str(model_path).endswith(".json"):
            # legacy XGBoost-only model file
            import xgboost as xgb
            self.model = xgb.XGBClassifier()
            self.model.load_model(model_path)
        else:
            self.model = joblib.load(model_path)

    def run(self, lab_values: dict) -> dict:
        fsh = lab_values["fsh"]
        lh = lab_values["lh"]
        amh = lab_values["amh"]
        fsh_lh_ratio = fsh / lh if lh != 0 else 0.0

        features = pd.DataFrame([{
            "fsh": fsh, "lh": lh, "fsh_lh_ratio": fsh_lh_ratio, "amh": amh
        }])

        prob = float(self.model.predict_proba(features)[0][1])
        return {
            "agent": "lab",
            "pcos_probability": prob,
            "predicted_label": "positive" if prob >= 0.5 else "negative",
            "fsh_lh_ratio": fsh_lh_ratio,
        }
