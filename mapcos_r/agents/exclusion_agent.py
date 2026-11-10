"""Exclusion Agent: thyroid dysfunction (TSH) and hyperprolactinaemia (prolactin).

No reference ranges are hard-coded. Ranges differ by laboratory/assay, so each result must come
with the reporting lab's own range (or the caller configures one via `reference_ranges`, which must
cite a source). Missing value or missing range -> UNKNOWN, never 'normal'.
Output is an AUTOMATED FLAG for a clinician, never a confirmed alternative diagnosis.
Conditions we cannot assess are listed explicitly (matches the architecture diagram).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from mapcos.schemas import ExclusionStatus

CHECKS = {
    "tsh": {"condition": "thyroid dysfunction", "why": "abnormal TSH can cause menstrual disturbance mimicking PCOS; guideline advises TSH screening before diagnosing"},
    "prolactin": {"condition": "hyperprolactinaemia", "why": "raised prolactin causes oligo/anovulation; guideline advises prolactin screening before diagnosing"},
}
NOT_ASSESSED = ["non-classic congenital adrenal hyperplasia (17-OHP)", "Cushing's syndrome",
                "androgen-secreting tumours", "other causes needing clinical review"]


def _num(v) -> Optional[float]:
    try:
        f = float(v)
        return None if f != f else f
    except (TypeError, ValueError):
        return None


class ExclusionAgent:
    def __init__(self, reference_ranges: Optional[dict] = None):
        """reference_ranges: {"tsh": {"low":..,"high":..,"unit":..,"source":".."}, ...} (optional, must cite source)."""
        self.ranges = reference_ranges or {}

    def _check(self, key: str, entry: Any) -> dict:
        meta = CHECKS[key]
        base = {"test": key, "condition": meta["condition"], "rationale": meta["why"]}
        entry = entry if isinstance(entry, dict) else ({"value": entry} if entry is not None else {})
        val = _num(entry.get("value"))
        if val is None:
            return {**base, "status": ExclusionStatus.UNKNOWN_MISSING_VALUE.value, "value": None,
                    "message": f"{key.upper()} result missing; further clinical evaluation needed. Missing is NOT treated as normal."}
        rng = {**self.ranges.get(key, {}), **{k: entry[k] for k in ("low", "high", "unit", "source") if entry.get(k) is not None}}
        low, high = _num(rng.get("low")), _num(rng.get("high"))
        if low is None and high is None:
            return {**base, "status": ExclusionStatus.UNKNOWN_NO_REFERENCE_RANGE.value, "value": val,
                    "message": "value present but no laboratory reference range supplied; cannot interpret."}
        flag = (low is not None and val < low) or (high is not None and val > high)
        out = {**base, "value": val, "unit": rng.get("unit"), "reference_range": {"low": low, "high": high},
               "range_source": rng.get("source", "reporting laboratory")}
        if flag:
            out.update(status=ExclusionStatus.ABNORMAL_FLAG.value,
                       message=f"{key.upper()} outside the supplied range: possible {meta['condition']} -- automated flag, "
                               "NOT a confirmed diagnosis; clinical evaluation recommended before attributing findings to PCOS.")
        else:
            out.update(status=ExclusionStatus.WITHIN_SUPPLIED_RANGE.value,
                       message=f"{key.upper()} within the supplied range (automated flag only).")
        return out

    def run(self, exclusion_labs: Optional[dict] = None) -> Dict[str, Any]:
        labs = exclusion_labs or {}
        checks = [self._check(k, labs.get(k)) for k in CHECKS]
        st = [c["status"] for c in checks]
        flagged = [c["condition"] for c in checks if c["status"] == ExclusionStatus.ABNORMAL_FLAG.value]
        unknown = [c["test"] for c in checks if c["status"].startswith("UNKNOWN")]
        if flagged:
            summary = "FLAGGED: " + ", ".join(flagged) + " may explain the presentation; clinician review required."
        elif unknown:
            summary = "INCOMPLETE: exclusion screening cannot be considered done (missing: " + ", ".join(unknown) + ")."
        else:
            summary = "No flag raised for TSH/prolactin against the supplied ranges. Other exclusions remain unassessed."
        return {"agent": "exclusion", "checks": checks, "flagged_conditions": flagged,
                "unknown_tests": unknown, "exclusion_screen_complete": not flagged and not unknown,
                "not_assessed": NOT_ASSESSED, "summary": summary, "is_confirmed_diagnosis": False}
