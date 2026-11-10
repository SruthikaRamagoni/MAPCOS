"""
Demo entry point.

Runs the pipeline built so far: CNN Classification Agent + Follicle Counting
Agent (in parallel) -> Grounding Agent. Lab, Symptoms, Synthesis, and
Exclusion agents are left as None; pass them in once implemented and the
orchestrator will pick them up automatically (see master_orchestrator.py).

Run: python -m mapcos.main <path_to_ultrasound_image>
"""

import sys
import tensorflow as tf

from mapcos.config import CNN_MODEL_PATH
from mapcos.agents.cnn_classification_agent import CNNClassificationAgent
from mapcos.agents.follicle_counting_agent import FollicleCountingAgent
from mapcos.agents.grounding_agent import GroundingAgent
from mapcos.orchestrator.master_orchestrator import MasterOrchestrator


def main(image_path: str):
    cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)

    orchestrator = MasterOrchestrator(
        cnn_agent=CNNClassificationAgent(cnn_model),
        follicle_agent=FollicleCountingAgent(),
        grounding_agent=GroundingAgent(),
    )

    result = orchestrator.run({"image_path": image_path})

    print(f"PCOS probability : {result['cnn_classification']['pcos_probability']:.3f}")
    print(f"Predicted label  : {result['cnn_classification']['predicted_label']}")
    print(f"Follicle count   : {result['follicle_counting']['follicle_count']}")
    print(f"Agreement status : {result['grounding']['agreement_status']}")
    print(f"Overlap (IoU)    : {result['grounding']['overlap_score']:.3f}")
    print(f"Reasoning        : {result['grounding']['reasoning']}")


if __name__ == "__main__":
    image_arg = sys.argv[1] if len(sys.argv) > 1 else "sample.jpg"
    main(image_arg)
