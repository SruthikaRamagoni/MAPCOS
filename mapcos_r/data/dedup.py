"""Duplicate detection + leakage-safe splitting for the ultrasound dataset.

Why: the original notebook only checked that files could be opened. Many
images in the Kaggle set are copies/augmentations of the same scan, so a random
split puts near-identical pictures in train AND test (data leakage).

Method
  1. Exact duplicates  -> SHA-256 of the file bytes.
  2. Near duplicates   -> 64-bit difference hash (dHash) of a 9x8 grayscale
                          thumbnail; two images are "near" if Hamming distance <= max_hamming.
  3. Union-find joins every duplicate/near-duplicate into a GROUP. Splitting is
     done by GROUP, so a group never straddles train/val/test.
No count is forced: whatever comes out is what gets reported.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import shutil
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash(path: str, size: int = 8) -> int:
    """64-bit difference hash; robust to resize / mild recompression."""
    with Image.open(path) as im:
        g = im.convert("L").resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


class _UF:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def scan_dataset(root: str, classes: Sequence[str], splits: Sequence[str] = ("train", "test")) -> List[dict]:
    """List readable images; unreadable ones are recorded, not silently lost."""
    rows: List[dict] = []
    for split in splits:
        for cls in classes:
            d = os.path.join(root, split, cls)
            if not os.path.isdir(d):
                continue
            for fn in sorted(os.listdir(d)):
                if os.path.splitext(fn)[1].lower() not in IMG_EXT:
                    continue
                p = os.path.join(d, fn)
                try:
                    with Image.open(p) as im:
                        im.load()
                    rows.append({"path": p, "label": cls, "orig_split": split, "readable": True})
                except Exception:
                    rows.append({"path": p, "label": cls, "orig_split": split, "readable": False})
    return rows


def find_groups(rows: List[dict], max_hamming: int = 4) -> List[dict]:
    """Annotate readable rows with sha256, dhash and a duplicate group id."""
    ok = [r for r in rows if r["readable"]]
    for r in ok:
        r["sha256"] = sha256_file(r["path"])
        r["dhash"] = dhash(r["path"])
    uf = _UF(len(ok))
    by_sha: Dict[str, int] = {}
    for i, r in enumerate(ok):
        if r["sha256"] in by_sha:
            uf.union(by_sha[r["sha256"]], i)
        else:
            by_sha[r["sha256"]] = i
    # Near-duplicate pass. Pigeonhole: with max_hamming=H, splitting the 64 bits into H+1
    # bands guarantees any pair within H flipped bits is identical in at least one band, so
    # bucketing by band finds every candidate pair; each is then verified by true Hamming distance.
    n_bands = max_hamming + 1
    edges = [round(64 * k / n_bands) for k in range(n_bands + 1)]
    bands: Dict[tuple, List[int]] = defaultdict(list)
    for i, r in enumerate(ok):
        for b in range(n_bands):
            width = edges[b + 1] - edges[b]
            bands[(b, (r["dhash"] >> edges[b]) & ((1 << width) - 1))].append(i)
    for idxs in bands.values():
        for x in range(len(idxs)):
            for y in range(x + 1, len(idxs)):
                i, j = idxs[x], idxs[y]
                if hamming(ok[i]["dhash"], ok[j]["dhash"]) <= max_hamming:
                    uf.union(i, j)
    for i, r in enumerate(ok):
        r["group"] = uf.find(i)
    return rows


def cross_split_groups(rows: List[dict]) -> int:
    """How many duplicate groups span >1 original split (= measured leakage)."""
    g = defaultdict(set)
    for r in rows:
        if r.get("readable"):
            g[r["group"]].add(r["orig_split"])
    return sum(1 for s in g.values() if len(s) > 1)


def group_split(rows: List[dict], fractions=(0.70, 0.15, 0.15), seed: int = 42) -> List[dict]:
    """Assign each GROUP to train/val/test, stratified by label, keeping one
    representative per exact-duplicate set (dedup) but whole near-dup groups together."""
    rng = random.Random(seed)
    ok = [r for r in rows if r.get("readable")]
    # keep one file per sha256 (exact duplicate removal)
    seen, kept = set(), []
    for r in ok:
        if r["sha256"] in seen:
            r["split"] = "dropped_exact_duplicate"
            continue
        seen.add(r["sha256"])
        kept.append(r)
    groups: Dict[int, List[dict]] = defaultdict(list)
    for r in kept:
        groups[r["group"]].append(r)
    # a group's label = majority label (conflicting labels inside a group are reported)
    by_label: Dict[str, List[int]] = defaultdict(list)
    for gid, members in groups.items():
        labels = [m["label"] for m in members]
        by_label[max(set(labels), key=labels.count)].append(gid)
    names = ("train", "val", "test")
    for label, gids in by_label.items():
        rng.shuffle(gids)
        total = sum(len(groups[g]) for g in gids)
        targets = [f * total for f in fractions]
        counts = [0, 0, 0]
        for gid in gids:
            k = max(range(3), key=lambda i: targets[i] - counts[i])
            counts[k] += len(groups[gid])
            for m in groups[gid]:
                m["split"] = names[k]
    return rows


def summarize(rows: List[dict]) -> dict:
    ok = [r for r in rows if r.get("readable")]
    kept = [r for r in ok if r.get("split") in ("train", "val", "test")]
    conflict = defaultdict(set)
    for r in kept:
        conflict[r["group"]].add(r["label"])
    out = {
        "files_found": len(rows),
        "unreadable": len(rows) - len(ok),
        "readable": len(ok),
        "exact_duplicates_dropped": sum(1 for r in ok if r.get("split") == "dropped_exact_duplicate"),
        "unique_after_exact_dedup": len(kept),
        "near_duplicate_groups_spanning_original_splits": cross_split_groups(rows),
        "groups_with_conflicting_labels": sum(1 for s in conflict.values() if len(s) > 1),
        "final_counts": {s: sum(1 for r in kept if r["split"] == s) for s in ("train", "val", "test")},
        "final_class_counts": {
            s: {c: sum(1 for r in kept if r["split"] == s and r["label"] == c)
                for c in sorted({r["label"] for r in kept})}
            for s in ("train", "val", "test")},
    }
    return out


def materialize(rows: List[dict], out_root: str) -> None:
    """Create out_root/{train,val,test}/<class>/ with symlinks (copy fallback)."""
    shutil.rmtree(out_root, ignore_errors=True)
    for r in rows:
        if r.get("split") not in ("train", "val", "test"):
            continue
        d = os.path.join(out_root, r["split"], r["label"])
        os.makedirs(d, exist_ok=True)
        dst = os.path.join(d, f"{r['sha256'][:12]}_{os.path.basename(r['path'])}")
        try:
            os.symlink(r["path"], dst)
        except OSError:
            shutil.copy2(r["path"], dst)


def build_clean_split(data_dir: str, classes: Sequence[str], out_root: str, manifest_path: str,
                      max_hamming: int = 4, seed: int = 42) -> dict:
    rows = scan_dataset(data_dir, classes)
    find_groups(rows, max_hamming)
    group_split(rows, seed=seed)
    summary = summarize(rows)
    slim = [{k: v for k, v in r.items() if k != "dhash"} | ({"dhash": f"{r['dhash']:016x}"} if "dhash" in r else {})
            for r in rows]
    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump({"config": {"max_hamming": max_hamming, "seed": seed, "classes": list(classes),
                              "fractions": [0.7, 0.15, 0.15]},
                   "summary": summary, "files": slim}, f, indent=1)
    materialize(rows, out_root)
    log.info("dedup summary: %s", summary)
    return summary


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    from mapcos.config import DATA_DIR_VISION
    root = sys.argv[1] if len(sys.argv) > 1 else DATA_DIR_VISION
    print(json.dumps(build_clean_split(root, ["notinfected", "infected"], "pcos_dedup",
                                       "artifacts/split_manifest.json"), indent=2))
