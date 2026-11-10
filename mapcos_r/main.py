"""
Full MAPCOS pipeline demo — runs all five implemented agents through the
Master Orchestrator: CNN Classification, Follicle Counting, Grounding, Lab,
and Symptoms. Synthesis, Exclusion, guideline retrieval and the report are included.

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
from mapcos.agents.synthesis_agent import SynthesisAgent
from mapcos.agents.exclusion_agent import ExclusionAgent
from mapcos.orchestrator.master_orchestrator import MasterOrchestrator
from mapcos.rag.retriever import GuidelineRetriever
from mapcos.reports.report_generator import ReportGenerator


def load_agents():
    cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
    return MasterOrchestrator(
        cnn_agent=CNNClassificationAgent(cnn_model),
        follicle_agent=FollicleCountingAgent(),
        grounding_agent=GroundingAgent(),
        lab_agent=LabAgent(model_path=LAB_MODEL_PATH),
        symptoms_agent=SymptomsAgent(model_path=SYMPTOMS_MODEL_PATH),
        synthesis_agent=SynthesisAgent(),
        exclusion_agent=ExclusionAgent(),
        retriever=GuidelineRetriever.default(),
        report_generator=ReportGenerator(),
        artifact_dir="mapcos_output/artifacts",
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

    # NOTE: the image and the spreadsheet row come from DIFFERENT people (public datasets), so
    # modalities_linked stays False and the report says so. TSH/prolactin + lab reference ranges are not
    # supplied here, so the Exclusion Agent will correctly report them as UNKNOWN.
    result = orchestrator.run({
        "image_path": image_path,
        "modalities_linked": False,
        "tabular_data": {"lab_values": lab_values, "symptoms": symptoms},
    })
    os.makedirs("mapcos_output", exist_ok=True)
    paths = ReportGenerator().save(result["report"], "mapcos_output/report")
    print("Report written:", paths)

    if result["errors"]:
        print("Agent errors:", result["errors"])
    print(f"\nImage              : {image_path}")
    print(f"CNN label          : {result['cnn_classification']['predicted_label']} "
          f"(p={result['cnn_classification']['pcos_probability']:.3f})")
    print(f"Follicle count     : {result['follicle_counting']['follicle_count']}")
    print(f"Agreement status   : {result['grounding']['agreement_status']}")
    print(f"Overlap (IoU)      : {result['grounding']['overlap_score']:.3f}")
    print(f"Lab result         : {result['lab']}")
    print(f"Symptoms result    : {result['symptoms']}")
    print(f"Rotterdam          : {result['synthesis']['provisional_interpretation']}")
    print(f"Exclusion          : {result['exclusion']['summary']}")
    return result


if __name__ == "__main__":
    image_arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(image_arg)
