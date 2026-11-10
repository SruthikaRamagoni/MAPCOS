"""Rotterdam Synthesis Agent (decision support, NOT a diagnosis).

Organises evidence under the adult Rotterdam / 2023 international guideline framework:
  C1 ovulatory dysfunction   C2 clinical or biochemical hyperandrogenism   C3 polycystic ovarian morphology (PCOM)
Two of three are needed AFTER exclusion of other causes (handled by the Exclusion Agent).

Safety rules enforced here (each has a unit test):
  * A CNN label is never a criterion; it is listed as 'supporting_model_evidence'.
  * The whole-image follicle count is not a criterion (not per-ovary, not calibrated, not validated).
  * PCOM = MET only from clinician-confirmed measurements (follicles per ovary >= threshold,
    ovarian volume >= threshold) or an AMH result with an assay-specific cut-off that the caller supplies.
  * FSH/LH ratio is not a criterion. AMH is only an alternative PCOM marker in adults, never standalone.
  * Missing information -> UNKNOWN, never NOT_MET.
  * Hyperandrogenism is NOT inferred from pimples / skin darkening / hair loss / weight gain.
    Only a clinician-confirmed hirsutism / hyperandrogenism flag, an explicit biochemical flag, or the
    dataset's 'hair growth' flag (reported as a low-certainty proxy) can inform C2.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from mapcos.schemas import Status

# Thresholds come from the guideline text (2023 international guideline). They are parameters,
# not buried constants, and are echoed in every output.
DEFAULTS = {"follicles_per_ovary": 20, "ovarian_volume_ml": 10.0, "min_age_years_for_pcom_or_amh": 18}


def _bool(v) -> Optional[bool]:
    if v is None:
        return None
    try:
        if v != v:           # NaN
            return None
    except Exception:
        pass
    return bool(v)


class SynthesisAgent:
    def __init__(self, thresholds: Optional[dict] = None):
        self.t = {**DEFAULTS, **(thresholds or {})}

    # ---- criteria -------------------------------------------------------
    def _c1_ovulation(self, clin: dict, symptoms: Optional[dict]) -> dict:
        explicit = _bool(clin.get("oligo_anovulation"))
        if explicit is not None:
            return self._crit("ovulatory_dysfunction", Status.MET if explicit else Status.NOT_MET,
                              "clinician-confirmed oligo/anovulation flag", "clinician")
        if symptoms and _bool(symptoms.get("cycle_irregular")) is not None:
            irr = _bool(symptoms["cycle_irregular"])
            return self._crit("ovulatory_dysfunction", Status.MET if irr else Status.NOT_MET,
                              "patient-reported cycle irregularity (self-report, not ovulation testing)",
                              "self_report", certainty="low")
        return self._crit("ovulatory_dysfunction", Status.UNKNOWN, "no cycle information supplied", None)

    def _c2_androgen(self, clin: dict, symptoms: Optional[dict]) -> dict:
        bio = _bool(clin.get("biochemical_hyperandrogenism"))
        hir = _bool(clin.get("hirsutism"))
        if bio is True or hir is True:
            src = "biochemical hyperandrogenism flag" if bio else "clinician-confirmed hirsutism"
            return self._crit("hyperandrogenism", Status.MET, src, "clinician")
        proxy = _bool(symptoms.get("hair_growth")) if symptoms else None
        if bio is False and hir is False:
            return self._crit("hyperandrogenism", Status.NOT_MET, "clinician reports neither sign", "clinician")
        if proxy is True:
            return self._crit("hyperandrogenism", Status.MET,
                              "dataset 'hair growth' flag used as a proxy for hirsutism (unconfirmed)",
                              "self_report", certainty="low")
        # Other symptoms (acne, skin darkening, hair loss, weight gain) are deliberately NOT used.
        if proxy is False and bio is None and hir is None:
            return self._crit("hyperandrogenism", Status.UNKNOWN,
                              "no reported hair growth, but hirsutism scoring and androgen blood tests are "
                              "missing, so absence cannot be concluded", None)
        return self._crit("hyperandrogenism", Status.UNKNOWN, "no hirsutism / androgen information supplied", None)

    def _c3_pcom(self, clin: dict, age: Optional[float]) -> dict:
        if age is not None and age < self.t["min_age_years_for_pcom_or_amh"]:
            return self._crit("polycystic_ovarian_morphology", Status.UNKNOWN,
                              "ultrasound/AMH are not used for diagnosis in adolescents under the guideline's adult criteria", None)
        fn = clin.get("follicles_per_ovary")
        vol = clin.get("ovarian_volume_ml")
        amh = clin.get("amh_result")  # {"value":..., "assay_cutoff":...}
        reasons: List[str] = []
        positive = False
        measured = False
        if fn is not None:
            measured = True
            ok = fn >= self.t["follicles_per_ovary"]
            positive |= ok
            reasons.append(f"follicle number per ovary {fn} ({'>=' if ok else '<'} {self.t['follicles_per_ovary']}), clinician-measured")
        if vol is not None:
            measured = True
            ok = vol >= self.t["ovarian_volume_ml"]
            positive |= ok
            reasons.append(f"ovarian volume {vol} mL ({'>=' if ok else '<'} {self.t['ovarian_volume_ml']} mL)")
        if amh and amh.get("value") is not None and amh.get("assay_cutoff") is not None:
            measured = True
            ok = amh["value"] >= amh["assay_cutoff"]
            positive |= ok
            reasons.append(f"AMH {amh['value']} vs caller-supplied assay-specific cut-off {amh['assay_cutoff']} (alternative marker, adults only)")
        elif amh and amh.get("value") is not None:
            reasons.append("AMH value present but no assay-specific cut-off supplied -> not used")
        if not measured:
            return self._crit("polycystic_ovarian_morphology", Status.UNKNOWN,
                              "no validated ultrasound measurement (per-ovary follicle count / volume) or AMH cut-off supplied; "
                              "CNN output and whole-image follicle count are not accepted as PCOM", None)
        if positive:
            return self._crit("polycystic_ovarian_morphology", Status.MET, "; ".join(reasons), "measurement")
        # a negative measurement is only 'NOT_MET' if it is a real measurement (not just an AMH-without-cutoff note)
        return self._crit("polycystic_ovarian_morphology", Status.NOT_MET, "; ".join(reasons), "measurement")

    @staticmethod
    def _crit(name, status, basis, source, certainty="standard") -> dict:
        return {"criterion": name, "status": status.value, "basis": basis, "source": source, "certainty": certainty}

    # ---- run ------------------------------------------------------------
    def run(self, cnn: Optional[dict] = None, follicle: Optional[dict] = None, grounding: Optional[dict] = None,
            lab: Optional[dict] = None, symptoms_result: Optional[dict] = None,
            symptoms_input: Optional[dict] = None, clinical_findings: Optional[dict] = None,
            age_years: Optional[float] = None) -> Dict[str, Any]:
        clin = clinical_findings or {}
        crit = [self._c1_ovulation(clin, symptoms_input), self._c2_androgen(clin, symptoms_input),
                self._c3_pcom(clin, age_years)]
        met = sum(c["status"] == "MET" for c in crit)
        notmet = sum(c["status"] == "NOT_MET" for c in crit)
        unknown = [c["criterion"] for c in crit if c["status"] == "UNKNOWN"]

        # supporting (non-criterion) model evidence
        support = []
        if cnn:
            support.append({"source": "cnn", "pcos_probability": cnn.get("pcos_probability"),
                            "note": "image classifier output; NOT confirmed polycystic ovarian morphology"})
        if follicle:
            support.append({"source": "follicle_counting", "whole_image_count": follicle.get("follicle_count"),
                            "note": "classical-CV whole-image count, not per ovary, not calibrated in mm, not validated"})
        if lab:
            support.append({"source": "lab_model", "pcos_probability": lab.get("pcos_probability"),
                            "note": "FSH/LH/AMH model; weak predictor in our hold-out (AUC~0.64); LH/FSH ratio is not a criterion"})
        if symptoms_result:
            support.append({"source": "symptoms_model", "pcos_probability": symptoms_result.get("pcos_probability"),
                            "rule_applied": symptoms_result.get("rule_applied"),
                            "note": "symptom model output; not a Rotterdam criterion"})

        conflicts = []
        if cnn and follicle:
            cnn_pos = cnn.get("pcos_probability", 0) >= 0.5
            fol_pos = follicle.get("follicle_count", 0) >= self.t["follicles_per_ovary"]
            if cnn_pos != fol_pos:
                conflicts.append("CNN and follicle-count agent point in different directions")
        if grounding and grounding.get("agreement_status") == "disagree":
            conflicts.append(f"grounding agent disagreement (IoU={grounding.get('overlap_score')})")
        if cnn and clin.get("follicles_per_ovary") is not None:
            if (cnn.get("pcos_probability", 0) >= 0.5) != (clin["follicles_per_ovary"] >= self.t["follicles_per_ovary"]):
                conflicts.append("CNN label conflicts with clinician-measured follicle number")
        if symptoms_result and lab and (symptoms_result.get("pcos_probability", 0) >= .5) != (lab.get("pcos_probability", 0) >= .5):
            conflicts.append("symptom model and lab model disagree")

        low = [c["criterion"] for c in crit if c["status"] == "MET" and c["certainty"] == "low"]
        if met >= 2:
            interp = ("Two or more Rotterdam criteria are provisionally MET on the supplied information. "
                      "Diagnosis additionally requires exclusion of other causes and clinician confirmation.")
            if low:
                interp += (" Caution: " + ", ".join(low) + " rest on self-reported/proxy data and need clinical confirmation.")
        elif notmet >= 2:
            interp = "At least two criteria are NOT_MET on the supplied information; Rotterdam threshold not reached."
        else:
            interp = ("INDETERMINATE: not enough confirmed information to decide whether two criteria are met "
                      f"({met} met, {notmet} not met, {len(unknown)} unknown).")
        c1, c2, c3 = crit
        note = None
        if c1["status"] == "MET" and c2["status"] == "MET":
            note = "Both clinical criteria met: the guideline does not require ultrasound or AMH for diagnosis in this situation."
        return {
            "agent": "synthesis", "framework": "Rotterdam 2-of-3 (2023 international guideline wording)",
            "criteria": crit, "criteria_met": met, "criteria_not_met": notmet, "unknown_criteria": unknown, "low_certainty_criteria": low,
            "supporting_model_evidence": support, "conflicting_evidence": conflicts,
            "guideline_note": note, "thresholds_used": self.t,
            "provisional_interpretation": interp, "requires_human_review": True,
            "is_definitive_diagnosis": False,
        }
