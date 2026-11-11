"""
Evaluates the Rotterdam 2-of-3 decision rule on REAL per-patient data.

The spreadsheet has, for the same patient, cycle regularity, hair growth and the
follicle count of each ovary, so all three criteria can be built without images.
This validates the rule logic against the PCOS (Y/N) label (the images are from
different people, so the image path cannot be validated per patient).

Caveat: the follicle counts here come from the spreadsheet, not from our follicle agent.
Run: python -m mapcos.training.evaluate_synthesis
"""

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from mapcos.agents.synthesis_agent import DiagnosticSynthesisAgent
from mapcos.config import (EXTRA_RAW_COLUMNS, SYMPTOMS_RAW_COLUMNS, TABULAR_SHEET_NAME,
                           TABULAR_XLSX_PATH, TARGET_COLUMN)


def main(thresholds=(12, 20)):
    df = pd.read_excel(TABULAR_XLSX_PATH, sheet_name=TABULAR_SHEET_NAME)
    need = [TARGET_COLUMN, EXTRA_RAW_COLUMNS["follicle_l"], EXTRA_RAW_COLUMNS["follicle_r"],
            SYMPTOMS_RAW_COLUMNS["cycle_code"], SYMPTOMS_RAW_COLUMNS["hair_growth"]]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found: {missing}\nAvailable: {list(df.columns)}")

    d = df[need].apply(pd.to_numeric, errors="coerce").dropna()
    y = d[TARGET_COLUMN].astype(int)
    rows = {}
    for thr in thresholds:
        agent = DiagnosticSynthesisAgent(pcom_threshold=thr)
        preds = []
        for _, r in d.iterrows():
            s = {"cycle_irregular": int(r[SYMPTOMS_RAW_COLUMNS["cycle_code"]] != 2),
                 "hair_growth": int(r[SYMPTOMS_RAW_COLUMNS["hair_growth"]])}
            pcom = max(r[EXTRA_RAW_COLUMNS["follicle_l"]], r[EXTRA_RAW_COLUMNS["follicle_r"]]) >= thr
            statuses = [agent._ovulatory(s)["status"], agent._androgen(s)["status"],
                        "met" if pcom else "not_met"]
            preds.append(int(agent.decide(statuses) == "PCOS likely"))
        tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
        rows[f"PCOM >= {thr} follicles"] = {
            "accuracy": accuracy_score(y, preds), "precision": precision_score(y, preds, zero_division=0),
            "recall": recall_score(y, preds, zero_division=0), "f1": f1_score(y, preds, zero_division=0),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}
    out = pd.DataFrame(rows).T
    print(f"Rotterdam rule on {len(d)} patients (positive = PCOS)")
    print(out.round(3))
    out.to_csv("synthesis_rule_evaluation.csv")
    return out


if __name__ == "__main__":
    main()
