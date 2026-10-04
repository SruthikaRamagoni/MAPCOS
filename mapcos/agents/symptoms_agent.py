"""
Symptoms Agent
----------------
Cycle regularity, hirsutism, BMI (+ related symptom flags) using the best
classifier chosen by training/train_symptoms_agent.py (Random Forest, XGBoost,
SVM or Decision Tree), plus a simple rule-based dampener for cases with no
androgenic signs at all.
"""

import joblib
import pandas as pd

FEATURE_ORDER = ["bmi", "cycle_irregular", "hair_growth", "weight_gain",
                 "skin_darkening", "hair_loss", "pimples"]


class SymptomsAgent:
    def __init__(self, model_path: str = None, model=None):
        if model is not None:
            self.model = model
        elif str(model_path).endswith(".json"):
            import xgboost as xgb
            self.model = xgb.XGBClassifier()
            self.model.load_model(model_path)
        else:
            self.model = joblib.load(model_path)

    def run(self, symptoms: dict) -> dict:
        features = pd.DataFrame([symptoms])[FEATURE_ORDER]
        prob = float(self.model.predict_proba(features)[0][1])

        no_androgenic_signs = (
            symptoms.get("hair_growth", 0) == 0 and
            symptoms.get("skin_darkening", 0) == 0 and
            symptoms.get("weight_gain", 0) == 0
        )
        rule_applied = False
        if no_androgenic_signs and prob > 0.4:
            prob = 0.4
            rule_applied = True

        return {
            "agent": "symptoms",
            "pcos_probability": prob,
            "predicted_label": "positive" if prob >= 0.5 else "negative",
            "rule_applied": rule_applied,
        }
