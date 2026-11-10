import numpy as np
from PIL import Image
from mapcos.data import dedup
from mapcos.evaluation.metrics import full_report


def _img(path, seed, size=(64, 64)):
    rng = np.random.RandomState(seed)
    base = rng.randint(0, 255, (16, 16), dtype=np.uint8)
    Image.fromarray(base).resize(size, Image.BICUBIC).convert("RGB").save(path)


def test_dedup_exact_near_and_leak_free_split(tmp_path):
    for cls in ("notinfected", "infected"):
        for sp in ("train", "test"):
            (tmp_path / sp / cls).mkdir(parents=True)
    # 6 distinct scans per class; each is copied into 'test' (exact) and as a resized copy (near)
    for cls in ("notinfected", "infected"):
        for k in range(6):
            seed = k + (100 if cls == "infected" else 0)
            _img(tmp_path / "train" / cls / f"a{k}.png", seed)
            _img(tmp_path / "test" / cls / f"b{k}.png", seed)                    # exact-content duplicate
            _img(tmp_path / "test" / cls / f"c{k}.png", seed, size=(96, 96))     # resized near-duplicate
    (tmp_path / "train" / "infected" / "broken.jpg").write_bytes(b"not an image")
    out = tmp_path / "clean"
    s = dedup.build_clean_split(str(tmp_path), ["notinfected", "infected"], str(out), str(tmp_path / "m.json"))
    assert s["unreadable"] == 1
    assert s["near_duplicate_groups_spanning_original_splits"] >= 12      # leakage measured in the raw layout
    assert sum(s["final_counts"].values()) == s["unique_after_exact_dedup"] < 36
    # leakage-free: no group appears in two splits
    import json
    files = [f for f in json.load(open(tmp_path / "m.json"))["files"] if f.get("split") in ("train", "val", "test")]
    g = {}
    for f in files:
        g.setdefault(f["group"], set()).add(f["split"])
    assert all(len(v) == 1 for v in g.values())
    assert s["groups_with_conflicting_labels"] == 0


def test_metrics_report():
    y = [0, 0, 1, 1, 1, 0]; p = [.1, .6, .8, .4, .9, .2]
    r = full_report(y, p)
    assert r["confusion_matrix"] == {"tn": 2, "fp": 1, "fn": 1, "tp": 2}
    assert abs(r["specificity"] - 2 / 3) < 1e-9 and 0 <= r["roc_auc"] <= 1 and r["calibration_bins"]


def test_metrics_single_class_auc_nan():
    import math
    assert math.isnan(full_report([1, 1], [.9, .8])["roc_auc"])
