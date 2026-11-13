"""
Safety / Human-in-the-Loop Agent
--------------------------------
Scores how risky it would be to release an automated result without a clinician looking
at it, and routes the case:

    LOW    -> AUTO_RELEASE_WITH_DISCLAIMER   (still decision support, never a diagnosis)
    MEDIUM -> PHYSICIAN_REVIEW
    HIGH   -> PRIORITY_PHYSICIAN_REVIEW      (hard stops: exclusion flag, inconclusive)

It does not diagnose. Input is the normalised dict built by
mapcos.orchestrator.extended_pipeline.normalize(). ReviewQueue stores the cases that were
routed to a physician and records the physician's decision.
"""

import json
import os
import time


class SafetyAgent:
    def __init__(self, uncertain_band=(0.35, 0.65), min_iou=0.10, max_plausible_follicles=50,
                 min_image_side=150):
        self.lo, self.hi, self.min_iou = uncertain_band[0], uncertain_band[1], min_iou
        self.max_follicles, self.min_side = max_plausible_follicles, min_image_side

    def assess(self, n):
        reasons, score, hard = [], 0, False

        def add(code, pts, msg, hard_stop=False):
            nonlocal score, hard
            reasons.append({"code": code, "points": pts, "message": msg})
            score += pts
            hard = hard or hard_stop

        p, meets = n.get("cnn_prob"), n.get("meets_pcom")
        if p is None:
            add("CNN_MISSING", 2, "No CNN probability was found in the pipeline output.")
        elif self.lo < p < self.hi:
            add("CNN_UNCERTAIN", 2, f"CNN probability {p:.2f} is in the uncertain band.")
        if p is not None and meets is not None and ((p >= 0.5) != bool(meets)):
            add("CNN_FOLLICLE_DISAGREE", 2, "CNN and follicle count disagree on the ultrasound finding.")
        if n.get("pcom_confidence") == "borderline":
            add("FOLLICLE_BORDERLINE", 1, "Strict and relaxed follicle counts fall on opposite sides of the PCOM threshold.")
        fc = n.get("follicle_count")
        if fc is not None and fc >= self.max_follicles:
            add("FOLLICLE_IMPLAUSIBLE", 2, f"{fc:.0f} follicles in one ovary is implausibly high, so the detector may be counting noise.")
        side = n.get("image_short_side")
        if side is not None and side < self.min_side:
            add("LOW_RESOLUTION", 1, f"Image is only {side}px on its short side; follicle sizes are unreliable.")
        iou = n.get("iou")
        if iou is not None and iou < self.min_iou:
            add("LOW_GROUNDING", 1, f"Grad-CAM region overlaps follicle detections poorly (IoU {iou:.2f}).")
        if n.get("lab_prob") is None:
            add("LAB_MISSING", 1, "No lab agent result available.")
        if n.get("symptoms_prob") is None and n.get("symptoms_present") is False:
            add("SYMPTOMS_MISSING", 1, "No symptoms agent result available.")
        if n.get("lab_prob") is not None and p is not None:
            if (n["lab_prob"] >= 0.5) != (p >= 0.5) and abs(n["lab_prob"] - 0.5) > 0.15 and abs(p - 0.5) > 0.15:
                add("LAB_IMAGING_CONFLICT", 1, "Lab-based and image-based probabilities point in opposite directions.")
        if n.get("exclusion_flagged"):
            add("EXCLUSION_FLAG", 3, "TSH or prolactin is outside limits, so PCOS cannot be confirmed until other causes are ruled out.", True)
        elif n.get("exclusion_present") is False:
            add("EXCLUSION_MISSING", 1, "Exclusion check unavailable, so the result is provisional.")
        elif n.get("exclusion_partial"):
            add("EXCLUSION_PARTIAL", 1, "Only one of TSH / prolactin was measured, so the result is provisional.")
        dx = (n.get("diagnosis") or "").lower()
        if "inconclusive" in dx or "clinical review" in dx:
            add("SYNTHESIS_INCONCLUSIVE", 2, "Synthesis agent reported an inconclusive result.", True)

        if hard or score >= 4:
            level, route = "HIGH", "PRIORITY_PHYSICIAN_REVIEW"
        elif score >= 2:
            level, route = "MEDIUM", "PHYSICIAN_REVIEW"
        else:
            level, route = "LOW", "AUTO_RELEASE_WITH_DISCLAIMER"
        return {"risk_score": score, "risk_level": level, "route": route, "reasons": reasons,
                "note": "Decision support only. Not a diagnosis, even at low risk."}


class ReviewQueue:
    """Append-only JSONL log of cases routed to a physician and the physician's decisions."""

    def __init__(self, path="review_queue.jsonl"):
        self.path = path

    def _append(self, rec):
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")

    def _records(self):
        if not os.path.exists(self.path):
            return []
        with open(self.path, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def submit(self, case_id, safety, summary=None):
        """Queue a case if it needs a physician. Returns True if it was queued."""
        if safety["route"] == "AUTO_RELEASE_WITH_DISCLAIMER":
            return False
        self._append({"event": "submitted", "case_id": case_id, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                      "route": safety["route"], "risk_level": safety["risk_level"],
                      "reasons": [r["code"] for r in safety["reasons"]], "summary": summary or {}})
        return True

    def resolve(self, case_id, reviewer, decision, notes=""):
        """decision: 'confirm' (agree with the automated result), 'override', or 'more_tests'."""
        assert decision in ("confirm", "override", "more_tests"), "decision must be confirm / override / more_tests"
        self._append({"event": "resolved", "case_id": case_id, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                      "reviewer": reviewer, "decision": decision, "notes": notes})

    def pending(self):
        state = {}
        for r in self._records():
            state[r["case_id"]] = r
        return [r for r in state.values() if r["event"] == "submitted"]
