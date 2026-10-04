"""
Follicle Counting Agent
--------------------------
Segments follicles as dark, round blobs (Otsu threshold) and separates
touching follicles using a distance-transform + local-maxima watershed.
Every image is resized to a fixed canonical size first, so area/radius
thresholds mean the same thing regardless of the input image's native
resolution (this dataset's images vary a lot in size).
"""

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from mapcos.config import FOLLICLE_PARAMS, PCOM_THRESHOLD


class FollicleCountingAgent:
    def __init__(self, params: dict = None, pcom_threshold: int = PCOM_THRESHOLD):
        self.p = {**FOLLICLE_PARAMS, **(params or {})}
        self.pcom_threshold = pcom_threshold

    def run(self, image_path: str) -> dict:
        img_bgr_orig = cv2.imread(image_path)
        if img_bgr_orig is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        p = self.p

        # --- NEW: normalize to a fixed size before anything else ---
        canonical_size = p["canonical_size"]  # (width, height)
        img_bgr = cv2.resize(img_bgr_orig, canonical_size, interpolation=cv2.INTER_AREA)
        # --------------------------------------------------------------

        h, w = img_bgr.shape[:2]
        crop_h = int(h * p["crop_fraction"])
        img_cropped = img_bgr[:crop_h, :]
        gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)

        denoised = cv2.bilateralFilter(
            gray, d=p["bilateral_d"],
            sigmaColor=p["bilateral_sigma_color"],
            sigmaSpace=p["bilateral_sigma_space"]
        )

        otsu_val, _ = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = np.where(denoised < otsu_val, 255, 0).astype(np.uint8)

        kernel = np.ones((3, 3), np.uint8)
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        dist = ndi.distance_transform_edt(opened)
        coords = peak_local_max(dist, min_distance=p["min_peak_distance"], labels=opened)
        peak_mask = np.zeros_like(dist, dtype=bool)
        peak_mask[tuple(coords.T)] = True
        markers, _ = ndi.label(peak_mask)
        labels_ws = watershed(-dist, markers, mask=opened)

        locations = []
        annotated = img_cropped.copy()

        for label in np.unique(labels_ws):
            if label == 0:
                continue
            region_mask = np.uint8(labels_ws == label)
            area = cv2.countNonZero(region_mask)
            if area < p["min_area"] or area > p["max_area"]:
                continue

            contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            cnt = contours[0]

            bx, by, bw_, bh_ = cv2.boundingRect(cnt)
            margin = p["border_margin"]
            touches_border = (
                bx <= margin or by <= margin or
                (bx + bw_) >= (w - margin) or (by + bh_) >= (crop_h - margin)
            )
            if touches_border:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < p["min_circularity"]:
                continue

            (cx, cy), r = cv2.minEnclosingCircle(cnt)
            locations.append({"x": int(cx), "y": int(cy), "r": int(r)})
            cv2.circle(annotated, (int(cx), int(cy)), int(r), (0, 255, 0), 2)

        count = len(locations)
        return {
            "agent": "follicle_counting",
            "follicle_count": count,
            "locations": locations,
            "image_shape": img_cropped.shape[:2],
            "meets_pcom_threshold": count >= self.pcom_threshold,
            "annotated_image": annotated,
        }
