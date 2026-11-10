"""
Follicle Counting Agent
--------------------------
Segments follicles as dark, round blobs (Otsu threshold) and separates
touching follicles using a distance-transform + local-maxima watershed.
Thresholds (area, peak spacing, border margin) are computed as a FRACTION
of each image's own size rather than fixed pixel counts, so detection
stays consistent whether the input image is 225x225 or 1200x800 — no
resizing needed, so no detail is lost on larger images.
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
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        p = self.p
        h, w = img_bgr.shape[:2]
        crop_h = int(h * p["crop_fraction"])
        img_cropped = img_bgr[:crop_h, :]
        gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)

        # --- thresholds scaled to THIS image's own size ---
        img_area = crop_h * w
        short_side = min(crop_h, w)
        min_area = max(5, int(img_area * p["min_area_fraction"]))
        max_area = int(img_area * p["max_area_fraction"])
        min_peak_distance = max(3, int(short_side * p["min_peak_distance_fraction"]))
        border_margin = max(2, int(short_side * p["border_margin_fraction"]))
        # ----------------------------------------------------

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
        coords = peak_local_max(dist, min_distance=min_peak_distance, labels=opened)
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
            if area < min_area or area > max_area:
                continue

            contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            cnt = contours[0]

            bx, by, bw_, bh_ = cv2.boundingRect(cnt)
            touches_border = (
                bx <= border_margin or by <= border_margin or
                (bx + bw_) >= (w - border_margin) or (by + bh_) >= (crop_h - border_margin)
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
                "original_image_shape": (h, w),   # NEW — the full image's size, before cropping
                "meets_pcom_threshold": count >= self.pcom_threshold,
                "annotated_image": annotated,
            }
