"""
End-to-end behaviour check of the whole pipeline on many cases.

Each test-split ultrasound image is paired with a RANDOM tabular patient row that has the
same PCOS label (the two datasets are different people, so this cannot measure per-patient
accuracy). It shows how the system behaves: how often it commits to an answer, how often it
sends a case to a physician, and whether the safety agent flags the cases the system gets wrong.

Run: python -m mapcos.training.evaluate_pipeline [n_per_class]
Writes pipeline_evaluation.csv and pipeline_summary.csv
"""

import sys

import cv2
import numpy as np
import pandas as pd

from mapcos.agents.safety_agent import SafetyAgent
from mapcos.config import (EXTRA_RAW_COLUMNS, LAB_RAW_COLUMNS, SYMPTOMS_RAW_COLUMNS, TABULAR_SHEET_NAME,
                           TABULAR_XLSX_PATH, TARGET_COLUMN)
from mapcos.orchestrator.extended_pipeline import normalize

SPLITS_CSV = "cnn_comparison/splits.csv"


def row_to_inputs(row):
    lab = {k: row[LAB_RAW_COLUMNS[k]] for k in ("fsh", "lh", "amh")}
    sym = {"bmi": row[SYMPTOMS_RAW_COLUMNS["bmi"]],
           "cycle_irregular": int(row[SYMPTOMS_RAW_COLUMNS["cycle_code"]] != 2),
           **{k: row[SYMPTOMS_RAW_COLUMNS[k]] for k in ("hair_growth", "weight_gain", "skin_darkening",
                                                          "hair_loss", "pimples")}}
    exc = {k: pd.to_numeric(row[EXTRA_RAW_COLUMNS[k]], errors="coerce") for k in ("tsh", "prl")}
    return lab, sym, exc


def evaluate(orchestrator, images: pd.DataFrame, tabular: pd.DataFrame, n_per_class=30, seed=0):
    """images: columns filepath,label ('infected'/'notinfected'). tabular: the patient spreadsheet."""
    rng = np.random.default_rng(seed)
    safety, rows = SafetyAgent(), []
    for label, want in (("infected", 1), ("notinfected", 0)):
        pool = images[images["label"] == label]
        pool = pool[[cv2.imread(p) is not None for p in pool["filepath"]]]
        pats = tabular[pd.to_numeric(tabular[TARGET_COLUMN], errors="coerce") == want]
        for _, r in pool.sample(min(n_per_class, len(pool)), random_state=seed).iterrows():
            lab, sym, exc = row_to_inputs(pats.iloc[int(rng.integers(len(pats)))])
            res = orchestrator.run({"image_path": r["filepath"],
                                    "tabular_data": {"lab_values": lab, "symptoms": sym, "exclusion": exc}})
            n = normalize(res)
            s = safety.assess(n)
            rows.append({"label": label, "image": r["filepath"], "cnn_prob": n["cnn_prob"],
                         "follicles": n["follicle_count"], "agreement": n["agreement"],
                         "diagnosis": n["diagnosis"], "final": n["final_diagnosis"],
                         "risk": s["risk_level"], "route": s["route"],
                         "flags": ",".join(x["code"] for x in s["reasons"])})
    d = pd.DataFrame(rows)
    d["decisive"] = d["diagnosis"].isin(["PCOS likely", "PCOS unlikely"])
    d["pred_pos"] = d["diagnosis"] == "PCOS likely"
    d["wrong"] = d["decisive"] & (d["pred_pos"] != (d["label"] == "infected"))
    return d


def summarise(d: pd.DataFrame) -> pd.DataFrame:
    def block(g):
        dec = g[g["decisive"]]
        return pd.Series({
            "cases": len(g),
            "mean_follicles": round(g["follicles"].mean(), 1),
            "committed_to_answer": round(g["decisive"].mean(), 2),
            "answer_correct_when_committed": round(1 - dec["wrong"].mean(), 2) if len(dec) else np.nan,
            "sent_to_physician": round((g["route"] != "AUTO_RELEASE_WITH_DISCLAIMER").mean(), 2),
            "auto_released": round((g["route"] == "AUTO_RELEASE_WITH_DISCLAIMER").mean(), 2),
        })
    by_label = pd.DataFrame({lbl: block(g) for lbl, g in d.groupby("label")}).T
    overall = block(d).to_frame("all").T
    # does routing catch the mistakes? error rate among committed answers, by route
    dec = d[d["decisive"]]
    route = dec.groupby(dec["route"] == "AUTO_RELEASE_WITH_DISCLAIMER")["wrong"].agg(["count", "mean"]).round(2)
    route.index = ["reviewed by physician" if not i else "auto-released" for i in route.index]
    return pd.concat([by_label, overall]), route


def main(n_per_class: int = 30):
    from mapcos.main import load_agents                    # imports tensorflow, kept lazy
    splits = pd.read_csv(SPLITS_CSV)
    images = splits[splits["split"] == "test"]
    tabular = pd.read_excel(TABULAR_XLSX_PATH, sheet_name=TABULAR_SHEET_NAME)
    d = evaluate(load_agents(), images, tabular, n_per_class)
    d.to_csv("pipeline_evaluation.csv", index=False)
    table, route = summarise(d)
    table.to_csv("pipeline_summary.csv")
    pd.set_option("display.width", 200)
    print("NOTE: images and patient rows are paired by label only (different people).\n")
    print(table.to_string(), "\n\nError rate of committed answers by routing:\n", route.to_string())
    return d


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
