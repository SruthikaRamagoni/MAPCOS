"""Integration tests with stub agents (SYNTHETIC). Verifies orchestration, not clinical performance."""
import json
import numpy as np
from mapcos.demo import StubCNN, StubFollicle, StubGrounding, StubLab, StubSymptoms, build_orchestrator
from mapcos.orchestrator.master_orchestrator import MasterOrchestrator
from mapcos.reports.report_generator import ReportGenerator
import cv2

SYM = dict(bmi=25, cycle_irregular=1, hair_growth=1, weight_gain=0, skin_darkening=0, hair_loss=0, pimples=0)
LAB = dict(fsh=5.0, lh=8.0, amh=4.0)


def img(tmp_path):
    p = tmp_path / "x.png"; cv2.imwrite(str(p), np.zeros((20, 20, 3), np.uint8)); return str(p)


class Boom:
    def run(self, *a):
        raise RuntimeError("model exploded")


def test_full_pipeline_json_safe_and_report(tmp_path):
    o = build_orchestrator(str(tmp_path / "art"))
    r = o.run({"image_path": img(tmp_path), "tabular_data": {"lab_values": LAB, "symptoms": SYM}})
    json.dumps(r, default=str)  # no numpy arrays left
    assert r["errors"] == [] and r["synthesis"]["criteria_met"] == 2
    assert r["modalities_linked"] is False
    assert "gradcam_overlay" in r["artifacts"]
    md = ReportGenerator().to_markdown(r["report"])
    for must in ("NOT a medical diagnosis", "Rotterdam", "Missing information", "Human-review", "not the same patient" if False else "NOT verified"):
        assert must in md
    assert r["report"]["guideline_references"]


def test_missing_image_skips_vision_only(tmp_path):
    r = build_orchestrator().run({"tabular_data": {"lab_values": LAB, "symptoms": SYM}})
    assert r["cnn_classification"]["status"] == "skipped" and r["grounding"]["status"] == "skipped"
    assert r["symptoms"]["pcos_probability"] > 0 and r["errors"] == []


def test_missing_tabular_skips_not_negative(tmp_path):
    r = build_orchestrator().run({"image_path": img(tmp_path)})
    assert r["lab"]["status"] == "skipped" and "pcos_probability" not in r["lab"]
    assert all(c["status"] == "UNKNOWN" for c in r["synthesis"]["criteria"])
    assert any("lab agent" in m for m in r["report"]["missing_information"])


def test_invalid_lab_value_skipped(tmp_path):
    r = build_orchestrator().run({"tabular_data": {"lab_values": {"fsh": None, "lh": float("nan"), "amh": 1}, "symptoms": SYM}})
    assert r["lab"]["status"] == "skipped" and "fsh" in r["lab"]["reason"]


def test_agent_failure_isolated(tmp_path):
    o = MasterOrchestrator(Boom(), StubFollicle(), StubGrounding(), StubLab(), Boom())
    r = o.run({"image_path": img(tmp_path), "tabular_data": {"lab_values": LAB, "symptoms": SYM}})
    assert r["cnn_classification"]["status"] == "error" and r["symptoms"]["status"] == "error"
    assert r["lab"]["pcos_probability"] > 0 and r["follicle_counting"]["follicle_count"] == 9
    assert r["grounding"]["status"] == "skipped" and len(r["errors"]) == 2


def test_nonexistent_image_recorded(tmp_path):
    r = build_orchestrator().run({"image_path": str(tmp_path / "nope.png")})
    assert r["errors"] and r["cnn_classification"]["status"] == "error"


def test_conflicting_outputs_surface_in_report(tmp_path):
    r = build_orchestrator().run({"image_path": img(tmp_path), "tabular_data": {"lab_values": LAB, "symptoms": SYM}})
    assert r["report"]["conflicting_evidence"]


def test_bad_input_type():
    import pytest
    with pytest.raises(TypeError):
        build_orchestrator().run("not a dict")


def test_linked_flag_respected(tmp_path):
    r = build_orchestrator().run({"image_path": img(tmp_path), "modalities_linked": True})
    assert r["report"]["modalities_linked_to_same_patient"] is True
