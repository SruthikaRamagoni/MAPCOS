"""
Full MAPCOS run: orchestrator (all agents) + Safety / Human-in-the-Loop + guideline RAG +
patient explanation + per-patient reasoning graph + final report.

Run:  python -m mapcos.run_extended
      python -m mapcos.run_extended /path/to/ultrasound.jpg [case_id]

Uses the demo patient row from mapcos.main (the image and the row come from different
people, so this shows the flow of the system, not diagnostic accuracy).
"""

import sys

from mapcos.agents.safety_agent import ReviewQueue
from mapcos.main import default_image_path, demo_tabular_row, load_agents
from mapcos.orchestrator.extended_pipeline import run_full


def main(image_path: str = None, case_id: str = "demo", out_dir: str = "report_out"):
    image_path = image_path or default_image_path()
    orchestrator = load_agents()
    lab_values, symptoms, exclusion = demo_tabular_row()
    result = orchestrator.run({
        "image_path": image_path,
        "tabular_data": {"lab_values": lab_values, "symptoms": symptoms, "exclusion": exclusion},
    })
    full = run_full(result, case_id=case_id, out_dir=out_dir, queue=ReviewQueue())
    n, s = full["normalized"], full["safety"]
    print(f"\nImage            : {image_path}")
    print(f"CNN              : {n['cnn_label']} (p={n['cnn_prob']:.3f})")
    print(f"Follicles        : {n['follicle_count']:.0f} in the fullest ovary (threshold {n['pcom_threshold']:.0f}, {n['pcom_confidence']})")
    print(f"Grounding        : {n['agreement']} (IoU {n['iou']})")
    print(f"Lab / symptoms   : {n['lab_prob']} / {n['symptoms_prob']}")
    print(f"Automated result : {n['final_diagnosis']}")
    print(f"Safety           : {s['risk_level']} -> {s['route']}")
    for r in s["reasons"]:
        print(f"                   [{r['code']}] {r['message']}")
    print(f"Report           : {out_dir}/{case_id}_final_report.md")
    return full


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None, sys.argv[2] if len(sys.argv) > 2 else "demo")
