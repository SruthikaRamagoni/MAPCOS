"""Lazy exports: importing e.g. mapcos.agents.synthesis_agent must NOT pull in TensorFlow/OpenCV."""
import importlib

_MAP = {
    "CNNClassificationAgent": "cnn_classification_agent", "FollicleCountingAgent": "follicle_counting_agent",
    "GroundingAgent": "grounding_agent", "LabAgent": "lab_agent", "SymptomsAgent": "symptoms_agent",
    "SynthesisAgent": "synthesis_agent", "ExclusionAgent": "exclusion_agent",
}
__all__ = list(_MAP)


def __getattr__(name):
    if name in _MAP:
        return getattr(importlib.import_module(f"{__name__}.{_MAP[name]}"), name)
    raise AttributeError(name)
