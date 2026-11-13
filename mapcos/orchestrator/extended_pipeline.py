"""
Extended pipeline
-----------------
Takes the result dict of MasterOrchestrator.run() and adds the layers that sit on top of it:

  normalize()        reads the orchestrator's real output schema into one flat dict
  SafetyAgent        risk score and routing (auto-release / physician review)
  GuidelineRAGAgent  retrieves the guideline passages relevant to this case
  PatientExplainer   plain-language, template-based summary
  per-patient reasoning graph (networkx) saved as PNG and GraphML
  final report       <case>_final_report.md and <case>_final.json

Usage:  run_full(orchestrator_result, case_id="patient1", out_dir="report_out")
"""

import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from mapcos.agents.guideline_rag_agent import GuidelineRAGAgent
from mapcos.agents.patient_explainer_agent import PatientExplainerAgent
from mapcos.agents.safety_agent import ReviewQueue, SafetyAgent
from mapcos.config import PCOM_THRESHOLD

# Edit this list so it states what you have actually validated before you cite the report.
LIMITATIONS = [
    "Ultrasound and tabular datasets come from different patients, so per-patient fusion is not validated.",
    "CNN results are affected by dataset leakage and shortcut cues (image size, brightness).",
    "The follicle counter is classical image processing; check its counts against hand counts before relying on them.",
]

COLORS = {"met": "#f4a261", "not_met": "#a8dadc", "uncertain": "#ffe66d", "unknown": "#dddddd", "": "#ffffff"}


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def normalize(res: dict) -> dict:
    """Flat view of a MasterOrchestrator result (uses the real keys, no guessing)."""
    cnn, fol = res["cnn_classification"], res["follicle_counting"]
    grd = res.get("grounding") or {}
    lab, sym, syn, exc = res.get("lab"), res.get("symptoms"), res.get("synthesis"), res.get("exclusion")
    diagnosis = syn["diagnosis"] if syn else ""
    shape = fol.get("original_image_shape", fol.get("image_shape"))
    return {
        "cnn_prob": _f(cnn["pcos_probability"]),
        "cnn_label": cnn["predicted_label"],
        "follicle_count": _f(fol["follicle_count"]),
        "total_detected": fol.get("total_detected"),
        "per_ovary_counts": fol.get("per_ovary_counts"),
        "meets_pcom": bool(fol["meets_pcom_threshold"]),
        "pcom_confidence": fol.get("pcom_confidence"),
        "pcom_threshold": _f(fol.get("pcom_threshold", PCOM_THRESHOLD)),
        "image_short_side": min(shape) if shape else None,
        "iou": _f(grd.get("overlap_score")),
        "agreement": grd.get("agreement_status"),
        "lab_prob": _f(lab["pcos_probability"]) if lab else None,
        "symptoms_prob": _f(sym["pcos_probability"]) if sym else None,
        "symptoms_present": sym is not None,
        "exclusion_present": bool(exc) and exc["status"] != "not_assessed",
        "exclusion_partial": bool(exc) and exc["status"] == "partially_assessed",
        "exclusion_flagged": bool(exc and exc["flags"]),
        "diagnosis": diagnosis,
        "final_diagnosis": exc["final_diagnosis"] if exc else diagnosis,
        "synthesis_confidence": syn["confidence"] if syn else None,
    }


def criteria_status(res: dict) -> dict:
    """Rotterdam criteria statuses straight from the Synthesis Agent."""
    syn = res.get("synthesis")
    if not syn:
        return {"pcom": "unknown", "ovulatory": "unknown", "androgen": "unknown"}
    c = syn["criteria"]
    return {"pcom": c["PCOM"]["status"], "ovulatory": c["OvulatoryDysfunction"]["status"],
            "androgen": c["Hyperandrogenism"]["status"]}


def build_graph(n, crit, safety, passages):
    G = nx.DiGraph()

    def node(i, kind, label, status=""):
        G.add_node(i, kind=kind, label=label, status=status or "")

    node("CNN", "agent", f"CNN\np={n['cnn_prob']:.2f}" if n["cnn_prob"] is not None else "CNN\nn/a")
    node("FOL", "agent", f"Follicles\n{n['follicle_count']:.0f}" if n["follicle_count"] is not None else "Follicles\nn/a")
    node("LAB", "agent", f"Lab\np={n['lab_prob']:.2f}" if n["lab_prob"] is not None else "Lab\nn/a")
    node("SYM", "agent", f"Symptoms\np={n['symptoms_prob']:.2f}" if n["symptoms_prob"] is not None else "Symptoms\nn/a")
    node("EXC", "agent", "Exclusion\nFLAG" if n["exclusion_flagged"] else "Exclusion")
    node("GRD", "agent", f"Grounding\nIoU={n['iou']:.2f}" if n["iou"] is not None else "Grounding")
    node("PCOM", "criterion", "PCOM", crit["pcom"])
    node("OVU", "criterion", "Ovulatory\ndysfunction", crit["ovulatory"])
    node("AND", "criterion", "Hyper-\nandrogenism", crit["androgen"])
    node("SYN", "decision", "Synthesis")
    node("SAF", "decision", f"Safety\n{safety['risk_level']}")
    node("OUT", "decision", safety["route"].replace("_", "\n"))
    for a, b, r in [("CNN", "GRD", "heatmap"), ("FOL", "GRD", "circles"), ("CNN", "PCOM", "supports"),
                    ("FOL", "PCOM", "supports"), ("SYM", "OVU", "supports"), ("SYM", "AND", "supports"),
                    ("PCOM", "SYN", "criterion"), ("OVU", "SYN", "criterion"), ("AND", "SYN", "criterion"),
                    ("LAB", "SYN", "supporting"), ("EXC", "SYN", "exclusion"), ("GRD", "SAF", "agreement"),
                    ("SYN", "SAF", "result"), ("EXC", "SAF", "hard stop"), ("SAF", "OUT", "route")]:
        G.add_edge(a, b, relation=r)
    target = {"pcom": "PCOM", "ovulation": "OVU", "androgen": "AND", "exclusion": "EXC",
              "labs": "LAB", "safety": "SAF"}
    for p in passages[:4]:
        pid = f"P:{p['id']}"
        node(pid, "passage", p["id"])
        G.add_edge(pid, target.get(p["topic"], "SYN"), relation="guideline")
    return G


def draw_graph(G, path):
    pos = {"CNN": (0, 4), "FOL": (0, 3), "LAB": (0, 2), "SYM": (0, 1), "EXC": (0, 0),
           "GRD": (1, 4), "PCOM": (1, 3), "OVU": (1, 1.5), "AND": (1, 0.5),
           "SYN": (2, 2), "SAF": (3, 2), "OUT": (4, 2)}
    k = 0
    for nd in G.nodes:
        if nd.startswith("P:"):
            pos[nd] = (1.5 + 0.55 * k, -1.2)
            k += 1
    plt.figure(figsize=(14, 7))
    colors = [COLORS.get(d["status"], "#ffffff") if d["kind"] == "criterion"
              else {"agent": "#cfe2f3", "decision": "#d9ead3", "passage": "#eeeeee"}[d["kind"]]
              for _, d in G.nodes(data=True)]
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=3200, edgecolors="black")
    nx.draw_networkx_labels(G, pos, {k2: v["label"] for k2, v in G.nodes(data=True)}, font_size=7)
    guide = [e for e in G.edges if e[0].startswith("P:")]
    nx.draw_networkx_edges(G, pos, edgelist=[e for e in G.edges if e not in guide], arrows=True, node_size=3200)
    nx.draw_networkx_edges(G, pos, edgelist=guide, style="dashed", edge_color="gray", arrows=True, node_size=3200)
    plt.title("Per-patient reasoning graph (orange = met, blue = not met, yellow = uncertain, grey = unknown)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def run_full(result: dict, case_id: str = "case", out_dir: str = "report_out",
             queue: ReviewQueue = None) -> dict:
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    n = normalize(result)
    crit = criteria_status(result)
    safety = SafetyAgent().assess(n)
    passages = GuidelineRAGAgent().retrieve(n, safety)
    patient = PatientExplainerAgent().explain(n, crit, safety)

    G = build_graph(n, crit, safety, passages)
    draw_graph(G, out / f"{case_id}_graph.png")
    nx.write_graphml(G, out / f"{case_id}_graph.graphml")

    cnn, fol = result["cnn_classification"], result["follicle_counting"]
    imgs = []
    for key, src, name in [("gradcam_overlay", cnn, "gradcam"), ("annotated_image", fol, "follicles")]:
        if src.get(key) is not None:
            cv2.imwrite(str(out / f"{case_id}_{name}.png"), src[key])
            imgs.append(f"![{name}]({case_id}_{name}.png)")

    queued = queue.submit(case_id, safety, {"final_diagnosis": n["final_diagnosis"], "cnn_prob": n["cnn_prob"],
                                            "follicle_count": n["follicle_count"]}) if queue else False
    held = safety["route"] != "AUTO_RELEASE_WITH_DISCLAIMER"
    fmt = lambda v: "n/a" if v is None else f"{v:.2f}"
    syn = result.get("synthesis")
    md = [f"# MAPCOS final report: {case_id}", "> Decision support only. Not a diagnosis.", "",
          f"**Automated result: {n['final_diagnosis'] or 'n/a'}**", "", "## Clinician summary",
          f"- CNN: **{n['cnn_label']}** (probability {fmt(n['cnn_prob'])})",
          f"- Follicles in the fullest ovary: **{n['follicle_count']:.0f}** (PCOM threshold {n['pcom_threshold']:.0f}, "
          f"confidence {n['pcom_confidence'] or 'n/a'}; total detected {n['total_detected']}, per region {n['per_ovary_counts']})",
          f"- Grounding: **{n['agreement']}**, IoU {fmt(n['iou'])}",
          f"- Lab model probability: {fmt(n['lab_prob'])} | Symptoms model probability: {fmt(n['symptoms_prob'])}",
          f"- Synthesis: {n['diagnosis'] or 'n/a'} (confidence {n['synthesis_confidence']})",
          f"- Exclusion check flagged: {n['exclusion_flagged']}",
          f"- Rotterdam criteria: {crit}"]
    if syn:
        md.append(f"- {syn['reasoning']}")
    md += ["", " ".join(imgs), "", "## Safety / human-in-the-loop",
           f"- Risk: **{safety['risk_level']}** (score {safety['risk_score']}) -> **{safety['route']}**"
           + (" (queued for physician)" if queued else "")]
    md += [f"  - [{r['code']}] {r['message']}" for r in safety["reasons"]] or ["  - No flags raised."]
    md += ["", "## Guideline evidence retrieved"]
    md += [f"- **{p['id']}** ({p['source']}, score {p['score']}): {p['text']}" for p in passages]
    md += ["", "## Patient explanation" + (" (hold until a physician has reviewed the case)" if held else ""),
           patient, "", "## Reasoning graph", f"![graph]({case_id}_graph.png)", "", "## Limitations"]
    md += [f"- {x}" for x in LIMITATIONS]
    (out / f"{case_id}_final_report.md").write_text("\n".join(md), encoding="utf-8")
    payload = {"normalized": n, "criteria": crit, "safety": safety, "passages": passages}
    (out / f"{case_id}_final.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return {**payload, "dir": str(out), "queued": queued}
