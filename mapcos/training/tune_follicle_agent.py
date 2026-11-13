"""
Validate and tune the Follicle Counting Agent on YOUR hand-counted images.

CSV format (header required):
    image,true_count[,pixel_spacing_mm]
true_count = follicles you counted in the fullest ovary of that image.

Usage:
    python -m mapcos.training.tune_follicle_agent labels.csv            # evaluate defaults
    python -m mapcos.training.tune_follicle_agent labels.csv --tune 150 # random search, 150 trials

Split your CSV by patient first (tune on one file, report on a held-out file) so the
reported accuracy is not optimistic.
"""
import argparse, csv, json, random
import cv2
import numpy as np

from mapcos.agents.follicle_counting_agent import FollicleCountingAgentV3, DEFAULTS
from mapcos.config import PCOM_THRESHOLD

SEARCH_SPACE = {
    "z_thresh": (2.0, 6.0), "min_log_contrast": (0.15, 0.5), "min_sector_contrast": (0.05, 0.25),
    "max_disc_vs_fov": (0.75, 0.97), "min_circularity": (0.4, 0.75), "max_aspect": (1.5, 2.5),
    "denoise_h": (4, 14), "nms_fraction": (0.01, 0.04),
}


def load_rows(csv_path):
    rows = []
    with open(csv_path, newline="") as fh:
        for r in csv.DictReader(fh):
            img = cv2.imread(r["image"])
            if img is None:
                print("skipping unreadable:", r["image"]); continue
            sp = float(r["pixel_spacing_mm"]) if r.get("pixel_spacing_mm") else None
            rows.append((img, int(r["true_count"]), sp))
    return rows


def evaluate(params, rows, thr=PCOM_THRESHOLD):
    agent = FollicleCountingAgentV3(params, pcom_threshold=thr)
    pred = np.array([agent.run_array(img, sp)["follicle_count"] for img, _, sp in rows])
    true = np.array([t for _, t, _ in rows])
    err = pred - true
    pos, neg = true >= thr, true < thr
    return {
        "n": len(rows), "MAE": float(np.abs(err).mean()), "RMSE": float(np.sqrt((err ** 2).mean())),
        "bias": float(err.mean()), "within_2": float((np.abs(err) <= 2).mean()),
        "pcom_accuracy": float(((pred >= thr) == pos).mean()),
        "pcom_sensitivity": float((pred[pos] >= thr).mean()) if pos.any() else None,
        "pcom_specificity": float((pred[neg] < thr).mean()) if neg.any() else None,
    }


def tune(rows, trials, seed=0):
    rng = random.Random(seed)
    best_p, best = dict(DEFAULTS), evaluate({}, rows)
    for t in range(trials):
        p = {k: (rng.randint(int(lo), int(hi)) if k == "denoise_h" else rng.uniform(lo, hi))
             for k, (lo, hi) in SEARCH_SPACE.items()}
        m = evaluate(p, rows)
        if m["MAE"] < best["MAE"]:
            best_p, best = {**DEFAULTS, **p}, m
            print(f"trial {t}: MAE {m['MAE']:.2f}  within±2 {m['within_2']:.2f}")
    return best_p, best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv"); ap.add_argument("--tune", type=int, default=0)
    a = ap.parse_args()
    rows = load_rows(a.csv)
    print("defaults:", json.dumps(evaluate({}, rows), indent=2))
    if a.tune:
        params, m = tune(rows, a.tune)
        print("best:", json.dumps(m, indent=2))
        json.dump({k: v for k, v in params.items() if k in SEARCH_SPACE}, open("follicle_best_params.json", "w"), indent=2)
        print("saved follicle_best_params.json  ->  FollicleCountingAgentV3(params=json.load(...))")
