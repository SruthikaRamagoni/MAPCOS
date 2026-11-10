
"""
Diagnostic Synthesis Agent
--------------------------
Applies the Rotterdam 2-of-3 rule to the outputs of the other agents.

  Ovulatory dysfunction : irregular cycle (symptoms input)
  Hyperandrogenism      : hirsutism (hair growth). Acne / hair loss are 'weak'
                          evidence: listed, but they never satisfy the criterion.
  PCOM                  : CNN says PCOS AND follicle count >= threshold -> met;
                          they disagree -> 'uncertain'; neither -> not met.

The symptoms / lab model probabilities are NOT part of the 2-of-3 count (the lab
model is weak: FSH/LH/AMH are not androgens). They are reported as supporting
signals, and a disagreement lowers the confidence.
"""

from mapcos.config import PCOM_THRESHOLD
from mapcos.knowledge_graph.guideline_kg import GuidelineKG


class DiagnosticSynthesisAgent:
    def __init__(self, kg: GuidelineKG = None, pcom_threshold: int = PCOM_THRESHOLD):
        self.kg = kg or GuidelineKG()
        self.pcom_threshold = pcom_threshold

    # ---- one method per criterion ------------------------------------------------
    def _ovulatory(self, s: dict) -> dict:
        met = int(s.get("cycle_irregular", 0)) == 1
        return {"status": "met" if met else "not_met",
                "evidence": ["irregular cycle"] if met else []}

    def _androgen(self, s: dict) -> dict:
        if int(s.get("hair_growth", 0)) == 1:
            return {"status": "met", "evidence": ["hirsutism (excess hair growth)"]}
        weak = [n for n in ("pimples", "hair_loss") if int(s.get(n, 0)) == 1]
        evidence = [f"{n} (weak sign, not counted)" for n in weak]
        return {"status": "not_met", "evidence": evidence}

    def _pcom(self, cnn: dict, follicle: dict, grounding: dict = None) -> dict:
        cnn_pos = cnn["predicted_label"] == "infected"
        count = follicle["follicle_count"]
        count_pos = count >= self.pcom_threshold
        evidence = [f"CNN: {cnn['predicted_label']} (p={cnn['pcos_probability']:.2f})",
                    f"follicle count {count} ({'>=' if count_pos else '<'} {self.pcom_threshold})"]
        if grounding is not None and grounding["agreement_status"] == "disagree":
            evidence.append("grounding check: CNN attention and follicles disagree")
        if cnn_pos and count_pos:
            status = "met"
        elif cnn_pos != count_pos:
            status = "uncertain"
        else:
            status = "not_met"
        return {"status": status, "evidence": evidence}

    # ---- decision rule (reused by training/evaluate_synthesis.py) -----------------
    def decide(self, statuses: list) -> str:
        met = statuses.count("met")
        uncertain = statuses.count("uncertain")
        need = self.kg.min_criteria()
        if met >= need:
            return "PCOS likely"
        if met + uncertain >= need:
            return "Inconclusive - needs clinical review"
        return "PCOS unlikely"

    def run(self, cnn: dict, follicle: dict, grounding: dict, lab: dict = None,
            symptoms_result: dict = None, symptoms_input: dict = None) -> dict:
        s = symptoms_input or {}
        criteria = {
            "OvulatoryDysfunction": self._ovulatory(s),
            "Hyperandrogenism": self._androgen(s),
            "PCOM": self._pcom(cnn, follicle, grounding),
        }
        statuses = [c["status"] for c in criteria.values()]
        diagnosis = self.decide(statuses)
        met = statuses.count("met")
        uncertain = statuses.count("uncertain")
        agree = grounding["agreement_status"] == "agree"

        if diagnosis == "PCOS likely":
            confidence = "high" if (uncertain == 0 and agree) else "moderate"
        elif diagnosis == "PCOS unlikely":
            confidence = "high" if all(x == "not_met" for x in statuses) else "moderate"
        else:
            confidence = "low"

        signals, notes = {}, []
        if symptoms_result is not None:
            signals["symptoms_probability"] = symptoms_result["pcos_probability"]
            if diagnosis == "PCOS likely" and symptoms_result["predicted_label"] == "negative":
                notes.append("Symptoms model disagrees with the criteria-based result.")
                confidence = "moderate"
        if lab is not None:
            signals["lab_probability"] = lab["pcos_probability"]
            notes.append("Lab model is a weak predictor; shown for context only.")

        names = {"met": "met", "not_met": "not met", "uncertain": "uncertain"}
        reasoning = (f"{met} of 3 Rotterdam criteria met"
                     + (f", {uncertain} uncertain" if uncertain else "") + ". "
                     + "; ".join(f"{k}: {names[v['status']]}" for k, v in criteria.items()) + ".")
        return {
            "agent": "synthesis",
            "diagnosis": diagnosis,
            "criteria_met": met,
            "criteria": criteria,
            "confidence": confidence,
            "supporting_signals": signals,
            "notes": notes,
            "reasoning": reasoning,
        }
