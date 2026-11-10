"""End-to-end DEMO of the full pipeline on SYNTHETIC inputs (no TensorFlow / datasets / trained models needed).

The CNN / follicle / lab / symptoms agents here are clearly-labelled STUBS returning fixed numbers so the
Synthesis -> Exclusion -> RAG -> Report chain can be shown and tested. None of this is clinical validation.
Real run: python -m mapcos.main <image> (needs trained models, see README).

Run: python -m mapcos.demo [out_dir]
"""
import sys

import numpy as np

from mapcos.agents.exclusion_agent import ExclusionAgent
from mapcos.agents.synthesis_agent import SynthesisAgent
from mapcos.orchestrator.master_orchestrator import MasterOrchestrator
from mapcos.rag.retriever import GuidelineRetriever
from mapcos.reports.report_generator import ReportGenerator


class StubCNN:
    def run(self, path):
        return {"agent": "cnn_classification", "pcos_probability": 0.97, "predicted_label": "infected",
                "gradcam_overlay": np.zeros((32, 32, 3), np.uint8), "gradcam_heatmap": np.zeros((7, 7))}


class StubFollicle:
    def run(self, path):
        return {"agent": "follicle_counting", "follicle_count": 9, "meets_pcom_threshold": False,
                "annotated_image": np.zeros((32, 32, 3), np.uint8), "locations": []}


class StubGrounding:
    def run(self, c, f):
        return {"agent": "grounding", "agreement_status": "disagree", "overlap_score": 0.10,
                "reasoning": "CNN predicts 'infected' but follicle count 9 is below the image-level threshold."}


class StubLab:
    def run(self, v):
        return {"agent": "lab", "pcos_probability": 0.46, "predicted_label": "negative", "fsh_lh_ratio": v["fsh"] / v["lh"]}


class StubSymptoms:
    def run(self, v):
        return {"agent": "symptoms", "pcos_probability": 0.81, "predicted_label": "positive", "rule_applied": False}


def build_orchestrator(artifact_dir=None):
    return MasterOrchestrator(StubCNN(), StubFollicle(), StubGrounding(), StubLab(), StubSymptoms(),
                              SynthesisAgent(), ExclusionAgent(), GuidelineRetriever.default(),
                              ReportGenerator(), artifact_dir=artifact_dir)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "demo_output"
    import os
    os.makedirs(out, exist_ok=True)
    img = os.path.join(out, "synthetic_input.png")
    import cv2
    cv2.imwrite(img, np.zeros((64, 64, 3), np.uint8))
    orch = build_orchestrator(os.path.join(out, "artifacts"))
    res = orch.run({
        "case_id": "DEMO-SYNTHETIC-001", "image_path": img,
        "tabular_data": {
            "lab_values": {"fsh": 5.2, "lh": 9.8, "amh": 6.1},
            "symptoms": {"bmi": 27.0, "cycle_irregular": 1, "hair_growth": 1, "weight_gain": 1,
                         "skin_darkening": 0, "hair_loss": 0, "pimples": 1},
            "exclusion_labs": {"tsh": {"value": 2.1}, "prolactin": {"value": 14.0, "low": 4.0, "high": 25.0, "unit": "ng/mL",
                                                                    "source": "SYNTHETIC demo range - not clinical"}},
        }})
    paths = ReportGenerator().save(res["report"], os.path.join(out, "demo_report"))
    print(open(paths["markdown"], encoding="utf-8").read())
