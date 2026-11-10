"""Shared vocabulary for MAPCOS agent outputs.

Every Rotterdam criterion and every exclusion check uses the same tri-state so
that "we have no information" can never be confused with "negative".
"""
from enum import Enum


class Status(str, Enum):
    MET = "MET"
    NOT_MET = "NOT_MET"
    UNKNOWN = "UNKNOWN"          # missing / not assessable -- NEVER treated as negative


class ExclusionStatus(str, Enum):
    WITHIN_SUPPLIED_RANGE = "WITHIN_SUPPLIED_RANGE"   # automated flag only, not a diagnosis
    ABNORMAL_FLAG = "ABNORMAL_FLAG"                   # automated flag only, not a diagnosis
    UNKNOWN_MISSING_VALUE = "UNKNOWN_MISSING_VALUE"
    UNKNOWN_NO_REFERENCE_RANGE = "UNKNOWN_NO_REFERENCE_RANGE"


DISCLAIMER = (
    "EXPERIMENTAL clinical decision-support output from a student research prototype. "
    "It is NOT a medical diagnosis and does not replace assessment by a qualified clinician."
)

UNLINKED_NOTE = (
    "The ultrasound image and the tabular (lab/symptom) record were NOT verified to come from "
    "the same patient. Multimodal fusion below is a software demonstration, not patient-level validation."
)
