"""
Full MAPCOS pipeline demo - runs every agent through the Master Orchestrator:
CNN Classification, Follicle Counting, Grounding, Lab, Symptoms, Diagnostic Synthesis
(Rotterdam 2-of-3), Exclusion (TSH / prolactin), then writes the clinical report.

Uses a test-split ultrasound image and the third row of the tabular dataset as a
demo patient. The image and the patient row come from different people, so this
demonstrates the flow of the system, not diagnostic accuracy.
For a specific image, pass its path as an argument.

Run: python -m mapcos.main
     python -m mapcos.main /path/to/some/ultrasound.jpg
"""

import os
import sys

import pandas as pd
import tensorflow as tf

from mapcos.config import (
    CNN_MODEL_PATH, DATA_DIR_VISION, LAB_MODEL_PATH, SYMPTOMS_MODEL_PATH,
    TABULAR_XLSX_PATH, TABULAR_SHEET_NAME, LAB_RAW_COLUMNS, SYMPTOMS_RAW_COLUMNS,
    EXTRA_RAW_COLUMNS,
)
from mapcos.agents.cnn_classification_agent import CNNClassificationAgent
from mapcos.agents.follicle_counting_agent import FollicleCountingAgent
from mapcos.agents.grounding_agent import GroundingAgent
from mapcos.agents.lab_agent import LabAgent
from mapcos.agents.symptoms_agent import SymptomsAgent
from mapcos.agents.synthesis_agent import DiagnosticSynthesisAgent
from mapcos.agents.exclusion_agent import ExclusionAgent
from mapcos.knowledge_graph.guideline_kg import GuidelineKG
from mapcos.orchestrator.master_orchestrator import MasterOrchestrator
from mapcos.reports.report_builder import build_report

REPORT_DIR = "report_out"
SPLITS_CSV = "cnn_comparison/splits.csv"     # written by training/compare_cnns.py


def load_agents():
    cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
    kg = GuidelineKG()
    return MasterOrchestrator(
        cnn_agent=CNNClassificationAgent(cnn_model),
        follicle_agent=FollicleCountingAgent(),
        grounding_agent=GroundingAgent(),
        lab_agent=LabAgent(model_path=LAB_MODEL_PATH),
        symptoms_agent=SymptomsAgent(model_path=SYMPTOMS_MODEL_PATH),
        synthesis_agent=DiagnosticSynthesisAgent(kg),
        exclusion_agent=ExclusionAgent(kg),
    )


def demo_tabular_row():
    df = pd.read_excel(TABULAR_XLSX_PATH, sheet_name=TABULAR_SHEET_NAME)
    row = df.iloc[2]
    lab_values = {
        "fsh": row[LAB_RAW_COLUMNS["fsh"]],
        "lh": row[LAB_RAW_COLUMNS["lh"]],
        "amh": row[LAB_RAW_COLUMNS["amh"]],
    }
    symptoms = {
        "bmi": row[SYMPTOMS_RAW_COLUMNS["bmi"]],
        "cycle_irregular": int(row[SYMPTOMS_RAW_COLUMNS["cycle_code"]] != 2),
        "hair_growth": row[SYMPTOMS_RAW_COLUMNS["hair_growth"]],
        "weight_gain": row[SYMPTOMS_RAW_COLUMNS["weight_gain"]],
        "skin_darkening": row[SYMPTOMS_RAW_COLUMNS["skin_darkening"]],
        "hair_loss": row[SYMPTOMS_RAW_COLUMNS["hair_loss"]],
        "pimples": row[SYMPTOMS_RAW_COLUMNS["pimples"]],
    }
    # Values the Exclusion Agent needs. Stray text becomes NaN, which counts as "not measured".
    exclusion = {
        "tsh": pd.to_numeric(row[EXTRA_RAW_COLUMNS["tsh"]], errors="coerce"),
        "prl": pd.to_numeric(row[EXTRA_RAW_COLUMNS["prl"]], errors="coerce"),
    }
    return lab_values, symptoms, exclusion


def default_image_path():
    """First TEST-split image of the leakage-safe split if it exists, otherwise the first
    image of the dataset's own test/infected folder."""
    if os.path.exists(SPLITS_CSV):
        splits = pd.read_csv(SPLITS_CSV)
        test = splits[(splits["split"] == "test") & (splits["label"] == "infected")]
        if len(test):
            return test.iloc[0]["filepath"]
    sample_dir = f"{DATA_DIR_VISION}/test/infected"
    return os.path.join(sample_dir, sorted(os.listdir(sample_dir))[0])


def main(image_path: str = None):
    if image_path is None:
        image_path = default_image_path()

    orchestrator = load_agents()
    lab_values, symptoms, exclusion = demo_tabular_row()

    result = orchestrator.run({
        "image_path": image_path,
        "tabular_data": {"lab_values": lab_values, "symptoms": symptoms, "exclusion": exclusion},
    })

    syn, exc = result["synthesis"], result["exclusion"]
    print(f"\nImage              : {image_path}")
    print(f"CNN label          : {result['cnn_classification']['predicted_label']} "
          f"(p={result['cnn_classification']['pcos_probability']:.3f})")
    print(f"Follicle count     : {result['follicle_counting']['follicle_count']}")
    print(f"Agreement status   : {result['grounding']['agreement_status']}")
    print(f"Overlap (IoU)      : {result['grounding']['overlap_score']:.3f}")
    print(f"Lab result         : {result['lab']}")
    print(f"Symptoms result    : {result['symptoms']}")
    print(f"Synthesis          : {syn['diagnosis']} (confidence: {syn['confidence']})")
    print(f"                     {syn['reasoning']}")
    print(f"Exclusion          : {exc['status']} -> {exc['final_diagnosis']}")

    report_path = build_report(result, out_dir=REPORT_DIR, case_id="demo")
    print(f"Report written to  : {report_path}")
    return result


if __name__ == "__main__":
    image_arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(image_arg)
