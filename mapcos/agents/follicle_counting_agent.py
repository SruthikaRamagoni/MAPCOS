"""
Follicle Counting Agent v2
--------------------------
Replaces the global-Otsu approach, which also counted the black background,
shadows and text as 'follicles'. Steps:

  1. Field of view: keep only the ultrasound area (largest bright region, eroded
     so the fan edge and overlays are ignored).
  2. Multi-scale difference-of-Gaussians: a follicle is a dark blob that is darker
     than its surroundings, at a scale relative to the image size.
  3. Local-maxima peaks above a response threshold.
  4. Contrast + position filter: the blob must be clearly darker than a ring
     around it, and that ring must lie inside the field of view.

Parameters are fractions of the image size, so any resolution works. Uses only
OpenCV + NumPy. Same output keys as FollicleCountingAgent, so it is a drop-in.
"""

import cv2
import numpy as np

from mapcos.config import PCOM_THRESHOLD

DEFAULTS = {
    "fov_thresh": 12,            # gray level above which a pixel counts as ultrasound area
    "fov_erode_fraction": 0.04,  # shrink the field of view by this fraction of the short side
    "sigma_fractions": (0.012, 0.018, 0.026, 0.036),  # blob scales (fraction of short side)
    "dog_k": 1.6,                # surround scale = k * centre scale
    "response_thresh": 5.0,      # min DoG response (gray levels) for a peak
    "min_contrast": 10.0,        # min (ring mean - disc mean) in gray levels
    "min_ring_in_fov": 0.8,      # fraction of the ring that must be inside the field of view
    "nms_fraction": 0.03,        # min distance between two follicles (fraction of short side)
    "max_follicles": 60,
}


class FollicleCountingAgentV2:
    def __init__(self, params: dict = None, pcom_threshold: int = PCOM_THRESHOLD):
        self.p = {**DEFAULTS, **(params or {})}
        self.pcom_threshold = pcom_threshold

    # -- step 1 -------------------------------------------------------------------
    def _field_of_view(self, gray):
        p = self.p
        short = min(gray.shape[:2])
        fov = (gray > p["fov_thresh"]).astype(np.uint8)
        k = max(3, int(short * 0.03) | 1)
        fov = cv2.morphologyEx(fov, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(fov)
        if n <= 1:
            return np.ones_like(gray, dtype=np.uint8)
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        fov = (lab == biggest).astype(np.uint8)
        flood = fov.copy()                      # fill holes (dark follicles inside the fan)
        ff = np.zeros((gray.shape[0] + 2, gray.shape[1] + 2), np.uint8)
        cv2.floodFill(flood, ff, (0, 0), 2)
        fov[flood != 2] = 1
        e = max(1, int(short * p["fov_erode_fraction"]))
        return cv2.erode(fov, np.ones((2 * e + 1, 2 * e + 1), np.uint8))

    # -- main ---------------------------------------------------------------------
    def run(self, image_path: str) -> dict:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return self.run_array(img)

    def run_array(self, img_bgr: np.ndarray) -> dict:
        p = self.p
        h, w = img_bgr.shape[:2]
        short = min(h, w)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        fov = self._field_of_view(gray)
        f = gray.astype(np.float32)

        # step 2: multi-scale DoG (positive where a pixel is darker than its surround)
        best = np.full(f.shape, -1e9, np.float32)
        best_sigma = np.zeros(f.shape, np.float32)
        for sf in p["sigma_fractions"]:
            s = max(1.5, sf * short)
            resp = cv2.GaussianBlur(f, (0, 0), s * p["dog_k"]) - cv2.GaussianBlur(f, (0, 0), s)
            upd = resp > best
            best[upd] = resp[upd]
            best_sigma[upd] = s

        # step 3: local maxima inside the field of view
        nms = max(3, int(short * p["nms_fraction"]) | 1)
        peaks = (best == cv2.dilate(best, np.ones((nms, nms), np.uint8))) \
                & (best > p["response_thresh"]) & (fov > 0)
        ys, xs = np.nonzero(peaks)
        order = np.argsort(-best[ys, xs])

        # step 4: contrast + ring-inside-FOV filter, then distance NMS
        yy, xx = np.mgrid[0:h, 0:w]
        kept = []
        for i in order:
            y, x = int(ys[i]), int(xs[i])
            r = max(3.0, float(best_sigma[y, x]) * np.sqrt(2))
            x0, x1 = max(0, int(x - 2.2 * r)), min(w, int(x + 2.2 * r) + 1)
            y0, y1 = max(0, int(y - 2.2 * r)), min(h, int(y + 2.2 * r) + 1)
            d2 = (xx[y0:y1, x0:x1] - x) ** 2 + (yy[y0:y1, x0:x1] - y) ** 2
            disc = d2 <= (0.7 * r) ** 2
            ring = (d2 >= (1.4 * r) ** 2) & (d2 <= (2.0 * r) ** 2)
            if disc.sum() < 3 or ring.sum() < 8:
                continue
            fov_patch = fov[y0:y1, x0:x1] > 0
            if fov_patch[ring].mean() < p["min_ring_in_fov"]:
                continue
            patch = f[y0:y1, x0:x1]
            if patch[ring & fov_patch].mean() - patch[disc].mean() < p["min_contrast"]:
                continue
            if any((x - kx) ** 2 + (y - ky) ** 2 < (0.8 * (r + kr)) ** 2 for kx, ky, kr in kept):
                continue
            kept.append((x, y, r))
            if len(kept) >= p["max_follicles"]:
                break

        annotated = img_bgr.copy()
        locations = []
        for x, y, r in kept:
            locations.append({"x": int(x), "y": int(y), "r": int(r)})
            cv2.circle(annotated, (int(x), int(y)), int(r), (0, 255, 0), 2)
        contours, _ = cv2.findContours(fov, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated, contours, -1, (255, 200, 0), 1)   # field of view outline

        count = len(locations)
        return {
            "agent": "follicle_counting",
            "follicle_count": count,
            "locations": locations,
            "image_shape": (h, w),
            "original_image_shape": (h, w),
            "meets_pcom_threshold": count >= self.pcom_threshold,
            "annotated_image": annotated,
        }
