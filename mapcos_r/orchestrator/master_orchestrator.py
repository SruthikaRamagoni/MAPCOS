"""Master Orchestrator.

Stage 1  CNN + Follicle (parallel) -> Grounding
Stage 2  Lab + Symptoms (parallel)
Stage 3  Synthesis (Rotterdam) -> Exclusion -> guideline retrieval -> Final report

Design rules
  * Constructor signature of the original orchestrator is preserved (new args are optional).
  * An agent exception never crashes the run: it is recorded in result["errors"] and that agent's
    slot becomes {"status": "error", ...}.
  * A missing modality (no image / no lab values ...) yields {"status": "skipped", "reason": ...} --
    never a made-up negative result.
  * Output is JSON-safe: numpy images are saved to `artifact_dir` and replaced by file paths.
  * `modalities_linked` defaults to False: the public datasets are from different people.
"""
from __future__ import annotations

import logging
import math
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional

import numpy as np

log = logging.getLogger(__name__)
LAB_KEYS = ("fsh", "lh", "amh")
SYMPTOM_KEYS = ("bmi", "cycle_irregular", "hair_growth", "weight_gain", "skin_darkening", "hair_loss", "pimples")


def _bad(v) -> bool:
    if v is None:
        return True
    try:
        return isinstance(v, float) and math.isnan(v) or (not isinstance(v, str) and float(v) != float(v))
    except (TypeError, ValueError):
        return True


def _missing_keys(d: Optional[dict], keys) -> list:
    d = d or {}
    return [k for k in keys if _bad(d.get(k))]


def _skip(agent: str, reason: str) -> dict:
    return {"agent": agent, "status": "skipped", "reason": reason}


class MasterOrchestrator:
    def __init__(self, cnn_agent, follicle_agent, grounding_agent,
                 lab_agent=None, symptoms_agent=None, synthesis_agent=None, exclusion_agent=None,
                 retriever=None, report_generator=None, artifact_dir: Optional[str] = None):
        self.cnn_agent, self.follicle_agent, self.grounding_agent = cnn_agent, follicle_agent, grounding_agent
        self.lab_agent, self.symptoms_agent = lab_agent, symptoms_agent
        self.synthesis_agent, self.exclusion_agent = synthesis_agent, exclusion_agent
        self.retriever, self.report_generator = retriever, report_generator
        self.artifact_dir = artifact_dir

    # -- safe execution --------------------------------------------------
    def _safe(self, name: str, fn: Callable, errors: list, *args) -> dict:
        try:
            return fn(*args)
        except Exception as e:  # noqa: BLE001 - isolation is the point
            log.exception("agent %s failed", name)
            errors.append({"agent": name, "error": f"{type(e).__name__}: {e}"})
            return {"agent": name, "status": "error", "error": f"{type(e).__name__}: {e}"}

    @staticmethod
    def _ok(res) -> bool:
        return bool(res) and res.get("status") not in ("skipped", "error")

    def _vision(self, image_path, errors):
        if not image_path:
            s = _skip("vision", "no ultrasound image supplied")
            return _skip("cnn_classification", s["reason"]), _skip("follicle_counting", s["reason"])
        if not os.path.isfile(image_path):
            r = f"image file not found"
            errors.append({"agent": "vision", "error": r})
            return ({"agent": "cnn_classification", "status": "error", "error": r},
                    {"agent": "follicle_counting", "status": "error", "error": r})
        with ThreadPoolExecutor(max_workers=2) as ex:
            cf = ex.submit(self._safe, "cnn_classification", self.cnn_agent.run, errors, image_path)
            ff = ex.submit(self._safe, "follicle_counting", self.follicle_agent.run, errors, image_path)
            return cf.result(), ff.result()

    def _tabular(self, tab: dict, errors: list):
        jobs = {}
        lab_in, sym_in = tab.get("lab_values"), tab.get("symptoms")
        out = {"lab": _skip("lab", "no lab agent configured") if self.lab_agent is None else None,
               "symptoms": _skip("symptoms", "no symptoms agent configured") if self.symptoms_agent is None else None}
        if self.lab_agent is not None:
            miss = _missing_keys(lab_in, LAB_KEYS)
            if miss:
                out["lab"] = _skip("lab", f"missing/invalid lab values: {', '.join(miss)}")
            else:
                jobs["lab"] = (self.lab_agent.run, lab_in)
        if self.symptoms_agent is not None:
            miss = _missing_keys(sym_in, SYMPTOM_KEYS)
            if miss:
                out["symptoms"] = _skip("symptoms", f"missing/invalid symptom values: {', '.join(miss)}")
            else:
                jobs["symptoms"] = (self.symptoms_agent.run, sym_in)
        if jobs:
            with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
                futs = {n: ex.submit(self._safe, n, fn, errors, arg) for n, (fn, arg) in jobs.items()}
                for n, f in futs.items():
                    out[n] = f.result()
        return out

    def _strip_arrays(self, result: dict, case_id: str) -> dict:
        """Save numpy images to disk, replace with paths -> JSON-safe result."""
        arts: Dict[str, str] = {}

        def save(name, arr):
            if self.artifact_dir is None:
                return None
            import cv2
            os.makedirs(self.artifact_dir, exist_ok=True)
            p = os.path.join(self.artifact_dir, f"{case_id}_{name}.png")
            cv2.imwrite(p, arr)
            return p
        for key, field, art in (("cnn_classification", "gradcam_overlay", "gradcam_overlay"),
                                ("follicle_counting", "annotated_image", "follicle_annotated")):
            res = result.get(key)
            if isinstance(res, dict):
                for f in list(res):
                    if isinstance(res[f], np.ndarray):
                        if f == field:
                            p = save(art, res[f])
                            if p:
                                arts[art] = p
                        del res[f]
        result["artifacts"] = arts
        return result

    def run(self, patient_input: dict) -> dict:
        """
        patient_input = {
          "case_id": optional str,
          "image_path": optional path,
          "tabular_data": {"lab_values": {...}, "symptoms": {...},
                           "exclusion_labs": {"tsh": {...}, "prolactin": {...}},      # optional
                           "clinical_findings": {...}, "age_years": float},           # optional
          "modalities_linked": bool (default False)
        }
        """
        if not isinstance(patient_input, dict):
            raise TypeError("patient_input must be a dict")
        case_id = str(patient_input.get("case_id") or uuid.uuid4().hex[:10])
        errors: list = []
        tab = patient_input.get("tabular_data") or {}
        cnn, fol = self._vision(patient_input.get("image_path"), errors)

        if self._ok(cnn) and self._ok(fol):
            grounding = self._safe("grounding", self.grounding_agent.run, errors, cnn, fol)
        else:
            grounding = _skip("grounding", "needs both CNN and follicle results")

        t = self._tabular(tab, errors)
        result: Dict[str, Any] = {"case_id": case_id, "modalities_linked": bool(patient_input.get("modalities_linked", False)),
                                  "cnn_classification": cnn, "follicle_counting": fol, "grounding": grounding,
                                  "lab": t["lab"], "symptoms": t["symptoms"], "synthesis": None, "exclusion": None,
                                  "guideline_retrieval": None, "errors": errors}

        ok = lambda r: r if self._ok(r) else None  # noqa: E731
        if self.synthesis_agent is not None:
            result["synthesis"] = self._safe(
                "synthesis", lambda: self.synthesis_agent.run(
                    cnn=ok(cnn), follicle=ok(fol), grounding=ok(grounding), lab=ok(t["lab"]),
                    symptoms_result=ok(t["symptoms"]), symptoms_input=tab.get("symptoms"),
                    clinical_findings=tab.get("clinical_findings"), age_years=tab.get("age_years")), errors)
        if self.exclusion_agent is not None:
            result["exclusion"] = self._safe("exclusion", self.exclusion_agent.run, errors, tab.get("exclusion_labs"))
        if self.retriever is not None:
            qs = ["Rotterdam criteria two of three diagnosis", "polycystic ovarian morphology follicle number ultrasound threshold",
                  "AMH alternative to ultrasound adults", "exclude thyroid prolactin before diagnosis"]
            result["guideline_retrieval"] = {q: self._safe("rag", self.retriever.retrieve, errors, q) for q in qs}

        result = self._strip_arrays(result, case_id)
        if self.report_generator is not None:
            result["report"] = self._safe("report", self.report_generator.build, errors, result)
        return result
