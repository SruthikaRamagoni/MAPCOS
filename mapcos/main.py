"""
Full MAPCOS pipeline demo — runs all five implemented agents through the
Master Orchestrator: CNN Classification, Follicle Counting, Grounding, Lab,
and Symptoms. Synthesis and Exclusion stay None until those are built.

Uses the first test-set ultrasound image and the first row of the tabular
dataset as a demo patient. For a specific image, pass its path as an argument.

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
)
from mapcos.agents.cnn_classification_agent import CNNClassificationAgent
from mapcos.agents.follicle_counting_agent import FollicleCountingAgent
from mapcos.agents.grounding_agent import GroundingAgent
from mapcos.agents.lab_agent import LabAgent
from mapcos.agents.symptoms_agent import SymptomsAgent
from mapcos.orchestrator.master_orchestrator import MasterOrchestrator


def load_agents():
    cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
    return MasterOrchestrator(
        cnn_agent=CNNClassificationAgent(cnn_model),
        follicle_agent=FollicleCountingAgent(),
        grounding_agent=GroundingAgent(),
        lab_agent=LabAgent(model_path=LAB_MODEL_PATH),
        symptoms_agent=SymptomsAgent(model_path=SYMPTOMS_MODEL_PATH),
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
    return lab_values, symptoms


def main(image_path: str = None):
    if image_path is None:
        sample_dir = f"{DATA_DIR_VISION}/test/infected"
        image_path = os.path.join(sample_dir, os.listdir(sample_dir)[0])

    orchestrator = load_agents()
    lab_values, symptoms = demo_tabular_row()

    result = orchestrator.run({
        "image_path": image_path,
        "tabular_data": {"lab_values": lab_values, "symptoms": symptoms},
    })

    print(f"\nImage              : {image_path}")
    print(f"CNN label          : {result['cnn_classification']['predicted_label']} "
          f"(p={result['cnn_classification']['pcos_probability']:.3f})")
    print(f"Follicle count     : {result['follicle_counting']['follicle_count']}")
    print(f"Agreement status   : {result['grounding']['agreement_status']}")
    print(f"Overlap (IoU)      : {result['grounding']['overlap_score']:.3f}")
    print(f"Lab result         : {result['lab']}")
    print(f"Symptoms result    : {result['symptoms']}")
    print(f"Synthesis result   : {result['synthesis']}   (not built yet)")
    print(f"Exclusion result   : {result['exclusion']}   (not built yet)")
    return result


if __name__ == "__main__":
    image_arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(image_arg)
