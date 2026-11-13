"""
Follicle Counting Agent v3
--------------------------
Classical CV (OpenCV + NumPy only), drop-in replacement for v2. It addresses the
weaknesses of v2:

  1. Per-ovary count    -> detections are grouped into ovary regions; `follicle_count`
                           is the count in the fullest region, because PCOM is defined
                           per ovary (the image-wide total is `total_detected`).
  2. Threshold choice   -> `pcom_threshold` is explicit (GUIDELINE_THRESHOLDS lists
                           12 for Rotterdam 2003 and 20 for the 2023 guideline).
  3. Adaptive thresholds-> everything runs on a log-intensity image (gain-invariant,
                           speckle is multiplicative) and the detection threshold is a
                           z-score against the image's own robust noise level.
  4. Size filtering     -> pass `pixel_spacing_mm` to keep only 2-9 mm follicles. Without
                           it, sizes are fractions of the image's short side.
  5. Real shapes        -> each candidate is segmented locally; ellipse axes,
                           circularity and aspect ratio are measured. Elongated dark
                           structures (vessels) and shapeless shadows are rejected.
  6. Validation         -> see training/tune_follicle_agent.py (MAE vs hand counts, PCOM
                           accuracy, and parameter tuning on your labelled images).

It also reports a strict and a relaxed count. If they fall on different sides of the
PCOM threshold, `pcom_confidence` is "borderline" so a human-review step can use it.

Output keys are a superset of v2's, so the Grounding and Synthesis agents work as is.
"""

import cv2
import numpy as np

from mapcos.config import PCOM_THRESHOLD

GUIDELINE_THRESHOLDS = {"rotterdam_2003": 12, "international_2023": 20}

DEFAULTS = {
    # field of view
    "fov_thresh": 12,             # gray level above which a pixel counts as ultrasound area
    "fov_erode_fraction": 0.02,   # centre of a follicle must be this far inside the fan
    # follicle size
    "min_diam_mm": 2.0,
    "max_diam_mm": 9.0,
    "min_diam_frac": 0.02,        # used when pixel spacing is unknown (fraction of short side)
    "max_diam_frac": 0.12,
    "n_scales": 6,
    "dog_k": 1.6,                 # surround scale = k * centre scale
    # detection, in the log-intensity domain
    "denoise_h": 8,               # non-local-means strength
    "z_thresh": 3.0,              # DoG response in units of the image's robust noise sigma
    "min_log_contrast": 0.30,     # ln(ring / disc): the follicle is >= ~26% darker than its ring
    "min_sector_contrast": 0.12,  # darkest-direction contrast; rejects one-sided edges
    "max_disc_vs_fov": 0.90,      # disc mean must be below 0.90 x median tissue brightness
    "min_ring_in_fov": 0.80,      # fraction of the ring that must lie inside the image area
    # shape (measured on the locally segmented blob)
    "min_circularity": 0.55,
    "max_aspect": 2.0,            # ellipse major / minor axis
    # housekeeping
    "nms_fraction": 0.02,
    "max_follicles": 80,
    "relax": 0.7,                 # relaxed pass multiplies thresholds by this
    "link_factor": 4.0,           # follicles closer than link_factor x median diameter share an ovary
}


class FollicleCountingAgentV3:
    def __init__(self, params: dict = None, pcom_threshold: int = PCOM_THRESHOLD):
        self.p = {**DEFAULTS, **(params or {})}
        self.pcom_threshold = pcom_threshold

    # ------------------------------------------------------------------ field of view
    def _field_of_view(self, gray):
        """Returns (fov_raw, fov_core): the ultrasound area and a slightly eroded copy."""
        p = self.p
        short = min(gray.shape[:2])
        fov = (gray > p["fov_thresh"]).astype(np.uint8)
        k = max(3, int(short * 0.03) | 1)
        fov = cv2.morphologyEx(fov, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(fov)
        if n <= 1:
            ones = np.ones_like(gray, dtype=np.uint8)
            return ones, ones
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        fov = (lab == biggest).astype(np.uint8)
        flood = fov.copy()                          # fill dark follicles inside the fan
        ff = np.zeros((gray.shape[0] + 2, gray.shape[1] + 2), np.uint8)
        cv2.floodFill(flood, ff, (0, 0), 2)
        fov[flood != 2] = 1
        e = max(1, int(short * p["fov_erode_fraction"]))
        core = cv2.erode(fov, np.ones((2 * e + 1, 2 * e + 1), np.uint8))
        return fov, core

    # ------------------------------------------------------------------ size range
    def _radius_range(self, short, pixel_spacing_mm):
        p = self.p
        if pixel_spacing_mm:
            d_min, d_max = p["min_diam_mm"] / pixel_spacing_mm, p["max_diam_mm"] / pixel_spacing_mm
        else:
            d_min, d_max = p["min_diam_frac"] * short, p["max_diam_frac"] * short
        return max(2.0, d_min), max(d_min * 1.5, d_max)

    # ------------------------------------------------------------------ candidate measurements
    def _measure(self, f, fov_raw, med_log, x, y, r, z):
        """Segments the blob around (x, y) and measures it. Returns a dict or None."""
        h, w = f.shape
        half = int(2.3 * r) + 1
        x0, x1 = max(0, x - half), min(w, x + half + 1)
        y0, y1 = max(0, y - half), min(h, y + half + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d2 = (xx - x) ** 2 + (yy - y) ** 2
        patch = f[y0:y1, x0:x1]
        in_fov = fov_raw[y0:y1, x0:x1] > 0

        disc = d2 <= (0.6 * r) ** 2
        ring = (d2 >= (1.5 * r) ** 2) & (d2 <= (2.1 * r) ** 2)
        if disc.sum() < 3 or ring.sum() < 8:
            return None
        ring_in_fov = float(in_fov[ring].mean())
        ring_vals = patch[ring & in_fov]
        if ring_vals.size < 8:
            return None
        disc_mean, ring_mean = float(patch[disc].mean()), float(ring_vals.mean())

        # blob-vs-edge test: a real follicle is darker than its ring in EVERY direction;
        # a shadow edge or fan border is only darker on one side.
        ang = np.arctan2(yy - y, xx - x)
        sector_min = 1e9
        for k in range(8):
            lo = -np.pi + k * np.pi / 4
            sec = ring & in_fov & (ang >= lo) & (ang < lo + np.pi / 4)
            if sec.sum() < 3:
                continue
            sector_min = min(sector_min, float(patch[sec].mean()) - disc_mean)
        if sector_min == 1e9:
            return None

        # local segmentation: pixels darker than the disc/ring midpoint, connected to the centre
        thr = float(np.log(0.5 * (np.exp(disc_mean) + np.exp(ring_mean))))   # midpoint in linear intensity
        dark = ((patch < thr) & (d2 <= (1.8 * r) ** 2) & in_fov).astype(np.uint8)
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab = cv2.connectedComponents(dark)
        cy, cx = y - y0, x - x0
        label = lab[cy, cx]
        if label == 0:
            return None
        blob = (lab == label).astype(np.uint8)
        cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        cnt = max(cnts, key=cv2.contourArea)
        area = float(cv2.contourArea(cnt))
        perim = float(cv2.arcLength(cnt, True))
        if area < 6 or perim <= 0:
            return None
        circ = min(1.0, 4 * np.pi * area / (perim ** 2))
        if len(cnt) >= 5:
            (_, _), (ax1, ax2), _ = cv2.fitEllipse(cnt)
            aspect = max(ax1, ax2) / max(1e-6, min(ax1, ax2))
        else:
            aspect = 1.0
        m = cv2.moments(cnt)
        gx = x0 + m["m10"] / max(m["m00"], 1e-6)
        gy = y0 + m["m01"] / max(m["m00"], 1e-6)
        diam = 2 * np.sqrt(area / np.pi)
        cnt_global = cnt + np.array([[[x0, y0]]], dtype=cnt.dtype)
        return {
            "x": gx, "y": gy, "diam": diam, "z": z,
            "log_contrast": ring_mean - disc_mean,
            "sector_min": sector_min,
            "disc_vs_fov": disc_mean - med_log,
            "ring_in_fov": ring_in_fov,
            "circ": circ, "aspect": aspect, "contour": cnt_global,
        }

    def _accept(self, m, s, d_min, d_max):
        """s = 1.0 strict; s < 1 relaxed (every test loosened)."""
        p = self.p
        return (
            m["z"] >= p["z_thresh"] * s
            and m["log_contrast"] >= p["min_log_contrast"] * s
            and m["sector_min"] >= p["min_sector_contrast"] * s
            and m["disc_vs_fov"] <= np.log(p["max_disc_vs_fov"]) * s
            and m["ring_in_fov"] >= p["min_ring_in_fov"] * s
            and m["circ"] >= p["min_circularity"] * s
            and m["aspect"] <= p["max_aspect"] / s
            and d_min * s <= m["diam"] <= d_max / s
        )

    @staticmethod
    def _nms(cands, short, frac):
        kept = []
        for m in sorted(cands, key=lambda c: -c["z"]):
            r = m["diam"] / 2
            if all((m["x"] - k["x"]) ** 2 + (m["y"] - k["y"]) ** 2 >= (0.8 * (r + k["diam"] / 2)) ** 2
                   and (m["x"] - k["x"]) ** 2 + (m["y"] - k["y"]) ** 2 >= (frac * short) ** 2
                   for k in kept):
                kept.append(m)
        return kept

    # ------------------------------------------------------------------ ovary grouping
    def _group(self, follicles, short):
        """Single-linkage clustering of follicle centres into ovary regions. Returns labels."""
        n = len(follicles)
        if n == 0:
            return []
        link = max(self.p["link_factor"] * float(np.median([f["diam"] for f in follicles])), 0.08 * short)
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(n):
            for j in range(i + 1, n):
                if (follicles[i]["x"] - follicles[j]["x"]) ** 2 + (follicles[i]["y"] - follicles[j]["y"]) ** 2 <= link ** 2:
                    parent[find(i)] = find(j)
        roots = {}
        return [roots.setdefault(find(i), len(roots)) for i in range(n)]

    # ------------------------------------------------------------------ main
    def run(self, image_path: str, pixel_spacing_mm: float = None) -> dict:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return self.run_array(img, pixel_spacing_mm)

    def run_array(self, img_bgr: np.ndarray, pixel_spacing_mm: float = None) -> dict:
        p = self.p
        h, w = img_bgr.shape[:2]
        short = min(h, w)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        fov_raw, fov_core = self._field_of_view(gray)

        # gain-invariant working image: denoise, then log (speckle is multiplicative)
        den = cv2.fastNlMeansDenoising(gray, None, h=p["denoise_h"],
                                       templateWindowSize=7, searchWindowSize=21)
        f = np.log(den.astype(np.float32) + 8.0)
        med_log = float(np.median(f[fov_raw > 0]))
        f[fov_raw == 0] = med_log                     # no fan-edge response

        # multi-scale DoG, each scale normalised by its own robust noise level
        d_min, d_max = self._radius_range(short, pixel_spacing_mm)
        radii = np.geomspace(d_min / 2, d_max / 2, p["n_scales"])
        best = np.full(f.shape, -1e9, np.float32)
        best_r = np.zeros(f.shape, np.float32)
        inside = fov_raw > 0
        for r in radii:
            s = max(1.2, r / np.sqrt(2))
            resp = cv2.GaussianBlur(f, (0, 0), s * p["dog_k"]) - cv2.GaussianBlur(f, (0, 0), s)
            vals = resp[inside]
            sigma_n = 1.4826 * np.median(np.abs(vals - np.median(vals))) + 1e-6
            z = resp / sigma_n
            upd = z > best
            best[upd], best_r[upd] = z[upd], r

        # peaks: relaxed threshold so the relaxed count has candidates to work with
        nms = max(3, int(short * p["nms_fraction"]) | 1)
        peaks = (best == cv2.dilate(best, np.ones((nms, nms), np.uint8))) \
                & (best > p["z_thresh"] * p["relax"]) & (fov_core > 0)
        ys, xs = np.nonzero(peaks)
        order = np.argsort(-best[ys, xs])[: 4 * p["max_follicles"]]

        cands = []
        for i in order:
            y, x = int(ys[i]), int(xs[i])
            r0, z0 = float(best_r[y, x]), float(best[y, x])
            m = self._measure(f, fov_raw, med_log, x, y, r0, z0)
            for _ in range(3):        # re-measure at the blob's own size if the DoG scale was off
                if m is None or abs(m["diam"] / 2 - r0) <= 0.2 * r0:
                    break
                r0 = float(np.clip(m["diam"] / 2, d_min / 4, d_max))
                m = self._measure(f, fov_raw, med_log, int(round(m["x"])), int(round(m["y"])), r0, z0)
            if m is not None:
                cands.append(m)

        self._last_candidates = cands                  # kept for debugging / tuning

        def select(s):
            ok = [m for m in cands if self._accept(m, s, d_min, d_max)]
            return self._nms(ok, short, p["nms_fraction"])[: p["max_follicles"]]

        strict, relaxed = select(1.0), select(p["relax"])

        def per_ovary(foll):
            labs = self._group(foll, short)
            counts = np.bincount(labs).tolist() if labs else []
            return labs, counts

        labs, counts = per_ovary(strict)
        count_strict = max(counts) if counts else 0
        _, rcounts = per_ovary(relaxed)
        count_relaxed = max(rcounts) if rcounts else 0

        thr = self.pcom_threshold
        borderline = (count_strict >= thr) != (count_relaxed >= thr)

        # outputs
        annotated = img_bgr.copy()
        locations = []
        palette = [(0, 255, 0), (0, 200, 255), (255, 0, 200), (255, 160, 0)]
        for m, lab in zip(strict, labs):
            r = max(1, int(round(m["diam"] / 2)))
            loc = {"x": int(round(m["x"])), "y": int(round(m["y"])), "r": r,
                   "ovary_id": int(lab), "diameter_px": round(float(m["diam"]), 1),
                   "circularity": round(float(m["circ"]), 2)}
            if pixel_spacing_mm:
                loc["diameter_mm"] = round(float(m["diam"]) * pixel_spacing_mm, 1)
            locations.append(loc)
            cv2.drawContours(annotated, [m["contour"]], -1, palette[int(lab) % len(palette)], 2)
        cnts, _ = cv2.findContours(fov_core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated, cnts, -1, (255, 200, 0), 1)
        cv2.putText(annotated, f"follicles: {count_strict} (ovary), {len(strict)} total",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        return {
            "agent": "follicle_counting",
            "follicle_count": count_strict,            # follicles in the fullest ovary region
            "total_detected": len(strict),
            "per_ovary_counts": counts,
            "count_strict": count_strict,
            "count_relaxed": count_relaxed,
            "locations": locations,
            "image_shape": (h, w),
            "original_image_shape": (h, w),
            "pcom_threshold": thr,
            "meets_pcom_threshold": count_strict >= thr,
            "pcom_confidence": "borderline" if borderline else "high",
            "size_filter": "mm" if pixel_spacing_mm else "fraction_of_image",
            "annotated_image": annotated,
        }


# Keep the original class names so agents/__init__.py and main.py keep working.
FollicleCountingAgentV2 = FollicleCountingAgentV3
FollicleCountingAgent = FollicleCountingAgentV3
