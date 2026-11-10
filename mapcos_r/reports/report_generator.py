"""Final Clinical Report: structured JSON + readable Markdown.

Combines every agent output, keeps missing information separate from negative findings and
always carries the experimental-decision-support disclaimer.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List

from mapcos.schemas import DISCLAIMER, UNLINKED_NOTE


def _p(x, nd=3):
    return "n/a" if x is None else (f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x))


def _agent_status(res):
    if res is None:
        return "NOT RUN / modality absent"
    if res.get("status") in ("skipped", "error"):
        return f"{res['status'].upper()}: {res.get('reason') or res.get('error')}"
    return "ok"


class ReportGenerator:
    def build(self, result: Dict[str, Any]) -> Dict[str, Any]:
        cnn, fol, gr = result.get("cnn_classification"), result.get("follicle_counting"), result.get("grounding")
        lab, sym = result.get("lab"), result.get("symptoms")
        syn, exc = result.get("synthesis"), result.get("exclusion")
        errors = result.get("errors", [])

        missing: List[str] = []
        for name, res in (("ultrasound CNN", cnn), ("follicle agent", fol), ("lab agent", lab), ("symptoms agent", sym)):
            if res is None or res.get("status") in ("skipped", "error"):
                missing.append(f"{name}: {_agent_status(res)}")
        if syn:
            missing += [f"Rotterdam criterion unknown: {c}" for c in syn["unknown_criteria"]]
        if exc:
            missing += [f"exclusion test not available: {t}" for t in exc["unknown_tests"]]
            missing += [f"not assessed by MAPCOS: {x}" for x in exc["not_assessed"]]

        review = ["Every output of this system requires clinician review."]
        if syn and syn["conflicting_evidence"]:
            review.append("Conflicting evidence present (see conflicts).")
        if exc and (exc["flagged_conditions"] or not exc["exclusion_screen_complete"]):
            review.append("Exclusion screening flagged or incomplete.")
        if syn and syn["unknown_criteria"]:
            review.append("Obtain confirmed clinical/ultrasound/lab information for UNKNOWN criteria.")
        if errors:
            review.append("One or more agents failed; results are partial.")

        guideline = result.get("guideline_retrieval") or {}
        uncertainty = [
            "Image CNN scores come from a small public dataset with possible duplicate leakage; treat as unvalidated.",
            "Lab model is a weak predictor in hold-out testing.",
            "Whole-image follicle count is not a clinical per-ovary count.",
        ]
        if not result.get("modalities_linked", False):
            uncertainty.insert(0, UNLINKED_NOTE)

        return {
            "case_id": result.get("case_id"), "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "disclaimer": DISCLAIMER,
            "modalities_linked_to_same_patient": bool(result.get("modalities_linked", False)),
            "ultrasound_cnn": None if not cnn or "pcos_probability" not in cnn else {
                "pcos_probability": cnn["pcos_probability"], "label": cnn["predicted_label"],
                "gradcam_reference": result.get("artifacts", {}).get("gradcam_overlay")},
            "follicle_analysis": None if not fol or "follicle_count" not in fol else {
                "whole_image_count": fol["follicle_count"], "meets_pcom_threshold_image_level": fol["meets_pcom_threshold"],
                "caveat": "not per ovary, not mm-calibrated, not validated; not used as a Rotterdam criterion",
                "annotated_image": result.get("artifacts", {}).get("follicle_annotated")},
            "grounding": gr, "laboratory": lab, "symptoms": sym,
            "rotterdam_assessment": syn, "exclusion_findings": exc,
            "guideline_references": guideline,
            "conflicting_evidence": (syn or {}).get("conflicting_evidence", []),
            "missing_information": missing, "uncertainty": uncertainty,
            "human_review_recommendations": review, "execution_errors": errors,
        }

    def to_markdown(self, rep: Dict[str, Any]) -> str:
        L: List[str] = []
        a = L.append
        a(f"# MAPCOS Clinical Decision-Support Report\n\n**Case:** `{rep['case_id']}`  \n**Generated (UTC):** {rep['generated_utc']}\n")
        a(f"> **{rep['disclaimer']}**\n")
        if not rep["modalities_linked_to_same_patient"]:
            a(f"> ⚠ {UNLINKED_NOTE}\n")
        a("## 1. Ultrasound (CNN)")
        c = rep["ultrasound_cnn"]
        a(f"- P(PCOS) = {_p(c['pcos_probability'])} → `{c['label']}`  \n- Grad-CAM: {c['gradcam_reference'] or 'not saved'}\n- *Image-classifier output only; it is not confirmed polycystic ovarian morphology.*" if c else "- Not available (no image or agent failed).")
        a("\n## 2. Follicle analysis")
        f = rep["follicle_analysis"]
        a(f"- Whole-image count: **{f['whole_image_count']}** (image-level ≥20 flag: {f['meets_pcom_threshold_image_level']})  \n- *{f['caveat']}*" if f else "- Not available.")
        a("\n## 3. Grounding (CNN vs follicles)")
        g = rep["grounding"]
        a(f"- {g['agreement_status'].upper()}, IoU = {_p(g['overlap_score'], 2)}  \n- {g['reasoning']}" if g and "agreement_status" in g else "- Not available.")
        a("\n## 4. Laboratory model")
        l = rep["laboratory"]
        a(f"- P(PCOS) = {_p(l['pcos_probability'])}, FSH/LH = {_p(l.get('fsh_lh_ratio'), 2)}  \n- *Supporting evidence only; LH/FSH ratio is not a Rotterdam criterion.*" if l and "pcos_probability" in l else f"- {_agent_status(l)}")
        a("\n## 5. Symptoms model")
        s = rep["symptoms"]
        a(f"- P(PCOS) = {_p(s['pcos_probability'])} (dampener applied: {s.get('rule_applied')})  \n- *Supporting evidence only.*" if s and "pcos_probability" in s else f"- {_agent_status(s)}")
        a("\n## 6. Rotterdam criterion assessment")
        r = rep["rotterdam_assessment"]
        if r:
            a("| Criterion | Status | Basis |\n|---|---|---|")
            for k in r["criteria"]:
                a(f"| {k['criterion']} | **{k['status']}** | {k['basis']} |")
            a(f"\n**{r['provisional_interpretation']}**")
            if r["guideline_note"]:
                a(f"\n{r['guideline_note']}")
        else:
            a("- Not available.")
        a("\n## 7. Exclusion findings (automated flags, not diagnoses)")
        e = rep["exclusion_findings"]
        if e:
            for k in e["checks"]:
                a(f"- **{k['test'].upper()}** – {k['status']}: {k['message']}")
            a(f"\n{e['summary']}")
        else:
            a("- Not run.")
        a("\n## 8. Guideline references")
        gr = rep["guideline_references"]
        if gr:
            for q, res in gr.items():
                a(f"**Query:** {q} → {res['status']}")
                for p in res["passages"][:2]:
                    ci = p["citation"]
                    a(f"- {p['text']}  \n  *[{ci['id']}] {ci['title']} ({ci['venue']}, {ci['date']}) {ci['url']}* — relevance {p['score']}")
                if res["message"]:
                    a(f"- {res['message']}")
        else:
            a("- None retrieved.")
        a("\n## 9. Conflicting evidence")
        a("\n".join(f"- {x}" for x in rep["conflicting_evidence"]) or "- None detected.")
        a("\n## 10. Missing information (NOT the same as negative)")
        a("\n".join(f"- {x}" for x in rep["missing_information"]) or "- None.")
        a("\n## 11. Confidence and uncertainty")
        a("\n".join(f"- {x}" for x in rep["uncertainty"]))
        a("\n## 12. Human-review recommendations")
        a("\n".join(f"- {x}" for x in rep["human_review_recommendations"]))
        if rep["execution_errors"]:
            a("\n## Execution errors")
            a("\n".join(f"- {x['agent']}: {x['error']}" for x in rep["execution_errors"]))
        a(f"\n---\n*{rep['disclaimer']}*")
        return "\n".join(L)

    def save(self, rep: Dict[str, Any], out_prefix: str) -> Dict[str, str]:
        jp, mp = out_prefix + ".json", out_prefix + ".md"
        with open(jp, "w") as f:
            json.dump(rep, f, indent=2, default=str)
        with open(mp, "w", encoding="utf-8") as f:
            f.write(self.to_markdown(rep))
        return {"json": jp, "markdown": mp}
