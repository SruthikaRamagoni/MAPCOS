"""Synthesis tests. All inputs are SYNTHETIC test fixtures, not clinical data."""
from mapcos.agents.synthesis_agent import SynthesisAgent

A = SynthesisAgent()
SYM = dict(bmi=25, cycle_irregular=1, hair_growth=1, weight_gain=0, skin_darkening=0, hair_loss=0, pimples=0)
st = lambda r, n: next(c["status"] for c in r["criteria"] if c["criterion"] == n)


def test_cnn_positive_alone_is_not_a_criterion():
    r = A.run(cnn={"pcos_probability": 0.99})
    assert r["criteria_met"] == 0 and st(r, "polycystic_ovarian_morphology") == "UNKNOWN"
    assert r["supporting_model_evidence"][0]["source"] == "cnn"


def test_whole_image_follicle_count_not_a_criterion():
    r = A.run(follicle={"follicle_count": 40})
    assert st(r, "polycystic_ovarian_morphology") == "UNKNOWN"


def test_all_missing_is_unknown_not_negative():
    r = A.run()
    assert all(c["status"] == "UNKNOWN" for c in r["criteria"]) and r["criteria_not_met"] == 0
    assert r["provisional_interpretation"].startswith("INDETERMINATE")


def test_two_clinical_criteria_met_and_note():
    r = A.run(symptoms_input=SYM)
    assert r["criteria_met"] == 2 and "provisionally MET" in r["provisional_interpretation"]
    assert "does not require ultrasound" in r["guideline_note"]


def test_acne_weight_gain_skin_do_not_imply_hyperandrogenism():
    s = {**SYM, "hair_growth": 0, "pimples": 1, "weight_gain": 1, "skin_darkening": 1, "hair_loss": 1}
    r = A.run(symptoms_input=s)
    assert st(r, "hyperandrogenism") == "UNKNOWN"


def test_regular_cycle_not_met():
    r = A.run(symptoms_input={**SYM, "cycle_irregular": 0})
    assert st(r, "ovulatory_dysfunction") == "NOT_MET"


def test_pcom_from_clinician_measurement():
    assert st(A.run(clinical_findings={"follicles_per_ovary": 22}), "polycystic_ovarian_morphology") == "MET"
    assert st(A.run(clinical_findings={"follicles_per_ovary": 12}), "polycystic_ovarian_morphology") == "NOT_MET"
    assert st(A.run(clinical_findings={"ovarian_volume_ml": 11}), "polycystic_ovarian_morphology") == "MET"


def test_amh_needs_assay_cutoff():
    r = A.run(clinical_findings={"amh_result": {"value": 9.0}})
    assert st(r, "polycystic_ovarian_morphology") == "UNKNOWN"
    r = A.run(clinical_findings={"amh_result": {"value": 9.0, "assay_cutoff": 4.0}})
    assert st(r, "polycystic_ovarian_morphology") == "MET"


def test_adolescent_pcom_not_used():
    r = A.run(clinical_findings={"follicles_per_ovary": 30}, age_years=15)
    assert st(r, "polycystic_ovarian_morphology") == "UNKNOWN"


def test_conflicts_detected():
    r = A.run(cnn={"pcos_probability": 0.95}, follicle={"follicle_count": 5},
              grounding={"agreement_status": "disagree", "overlap_score": 0.0},
              lab={"pcos_probability": 0.9}, symptoms_result={"pcos_probability": 0.1})
    assert len(r["conflicting_evidence"]) == 3


def test_never_definitive():
    r = A.run(symptoms_input=SYM)
    assert r["is_definitive_diagnosis"] is False and r["requires_human_review"] is True


def test_nan_inputs_are_unknown():
    r = A.run(symptoms_input={**SYM, "cycle_irregular": float("nan"), "hair_growth": float("nan")})
    assert st(r, "ovulatory_dysfunction") == "UNKNOWN" and st(r, "hyperandrogenism") == "UNKNOWN"


def test_low_certainty_flagged_in_interpretation():
    r = A.run(symptoms_input=SYM)
    assert set(r["low_certainty_criteria"]) == {"ovulatory_dysfunction", "hyperandrogenism"}
    assert "self-reported" in r["provisional_interpretation"]
    r2 = A.run(clinical_findings={"oligo_anovulation": True, "hirsutism": True})
    assert r2["low_certainty_criteria"] == []
