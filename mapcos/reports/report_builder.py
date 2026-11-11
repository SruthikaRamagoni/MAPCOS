
"""
Final Clinical Report
---------------------
Turns the orchestrator result into report.md (clinician section + plain-language
patient section) and saves the Grad-CAM and follicle images next to it.
Decision support only - it never states a diagnosis as final.
"""

import os

import cv2

DISCLAIMER = ("This report is produced by a research prototype for decision support. "
              "It is not a diagnosis; a qualified clinician must confirm any finding.")

PATIENT_TEXT = {
    "likely": "The tests suggest that PCOS is likely. Please discuss these results with your doctor, "
              "who will confirm the diagnosis and decide on next steps.",
    "unlikely": "The tests do not strongly suggest PCOS. If you still have symptoms, please speak to your doctor.",
    "inconclusive": "The results are unclear, so your doctor needs to review them and may order more tests.",
}


def _patient_line(final: str) -> str:
    f = final.lower()
    if "not confirmed" in f:
        return ("Some signs could also be caused by another condition (for example a thyroid or hormone "
                "problem). Your doctor should check this before any PCOS diagnosis.")
    if "likely" in f and "unlikely" not in f:
        return PATIENT_TEXT["likely"]
    if "unlikely" in f:
        return PATIENT_TEXT["unlikely"]
    return PATIENT_TEXT["inconclusive"]


def build_report(result: dict, out_dir: str = "report_out", case_id: str = "case") -> str:
    os.makedirs(out_dir, exist_ok=True)
    cnn, fol, grd = result["cnn_classification"], result["follicle_counting"], result["grounding"]
    syn, exc = result["synthesis"], result["exclusion"]

    cam_path = f"{case_id}_gradcam.png"
    fol_path = f"{case_id}_follicles.png"
    cv2.imwrite(os.path.join(out_dir, cam_path), cnn["gradcam_overlay"])
    cv2.imwrite(os.path.join(out_dir, fol_path), fol["annotated_image"])

    final = exc["final_diagnosis"] if exc else (syn["diagnosis"] if syn else "n/a")
    lines = [f"# MAPCOS Clinical Report - {case_id}", "", f"**Result: {final}**", ""]

    if syn:
        lines += ["## Rotterdam criteria (2 of 3 needed)", "",
                  "| Criterion | Status | Evidence |", "|---|---|---|"]
        for name, c in syn["criteria"].items():
            lines.append(f"| {name} | {c['status'].replace('_', ' ')} | {'; '.join(c['evidence']) or '-'} |")
        lines += ["", f"{syn['reasoning']} Criteria confidence (before exclusion check): **{syn['confidence']}**.", ""]
        for n in syn["notes"]:
            lines.append(f"- {n}")
        sig = syn["supporting_signals"]
        if sig:
            lines.append("- Model signals: " + ", ".join(f"{k} = {v:.2f}" for k, v in sig.items()))
        lines.append("")

    lines += ["## Ultrasound findings", "",
              f"- CNN: **{cnn['predicted_label']}** (probability {cnn['pcos_probability']:.2f})",
              f"- Follicles detected: **{fol['follicle_count']}**",
              f"- Agreement check: **{grd['agreement_status']}** (IoU {grd['overlap_score']:.2f})",
              f"- {grd['reasoning']}", "",
              f"![Grad-CAM]({cam_path})  ![Follicles]({fol_path})", ""]

    if exc:
        lines += ["## Other causes (exclusion)", "", f"Status: **{exc['status'].replace('_', ' ')}**", ""]
        for f in exc["flags"]:
            lines.append(f"- RAISED {f['marker']}: {f['value']} {f['unit']} (limit {f['upper_limit']}). {f['advice']}")
        if exc["clear"]:
            lines.append("- Within limits: " + ", ".join(exc["clear"]))
        if exc["not_assessed"]:
            lines.append("- Not measured: " + ", ".join(exc["not_assessed"]))
        lines += ["- Not assessable from this data: " + ", ".join(exc["not_assessable_from_data"]), ""]

    lines += ["## Summary for the patient", "", _patient_line(final), "", "---", f"*{DISCLAIMER}*", ""]
    path = os.path.join(out_dir, f"{case_id}_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path
