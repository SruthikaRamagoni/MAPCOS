"""
Agreement / Grounding Agent
------------------------------
Cross-modal sanity check between the CNN Classification Agent and the
Follicle Counting Agent:
  - Overlays Grad-CAM with detected follicle circles
  - Computes an IoU (intersection-over-union) score between the two
  - Compares follicle count with the PCOM threshold (>=20 per ovary)

Outputs: agreement status (agree/disagree) + overlap score + reasoning.
Note: the CNN agent's image_shape is its resized (e.g. 224x224) input;
the Follicle agent's image_shape is the cropped original image size —
this agent rescales follicle coordinates to match before comparing.
"""

import cv2
import numpy as np

from mapcos.config import GROUNDING_PARAMS, PCOM_THRESHOLD
from mapcos.utils.gradcam import heatmap_to_mask
from mapcos.utils.iou import compute_iou


class GroundingAgent:
    def __init__(self, params: dict = None, pcom_threshold: int = PCOM_THRESHOLD):
        self.p = {**GROUNDING_PARAMS, **(params or {})}
        self.pcom_threshold = pcom_threshold

    def _follicle_mask(self, locations: list, shape: tuple) -> np.ndarray:
        mask = np.zeros(shape, dtype=np.uint8)
        for loc in locations:
            cv2.circle(mask, (loc["x"], loc["y"]), loc["r"], 1, thickness=-1)
        return mask

    def run(self, cnn_result: dict, follicle_result: dict) -> dict:
        target_shape = cnn_result["image_shape"]  # (H, W) the Grad-CAM heatmap is sized to

        gradcam_mask = heatmap_to_mask(
            cnn_result["gradcam_heatmap"], target_shape,
            threshold=self.p["heatmap_threshold"]
        )

        orig_h, orig_w = follicle_result.get("original_image_shape", follicle_result["image_shape"])
        target_h, target_w = target_shape
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h
        scaled_locations = [
            {
                "x": int(loc["x"] * scale_x),
                "y": int(loc["y"] * scale_y),
                "r": max(1, int(loc["r"] * ((scale_x + scale_y) / 2))),
            }
            for loc in follicle_result["locations"]
        ]
        follicle_mask = self._follicle_mask(scaled_locations, target_shape)

        overlap_score = compute_iou(gradcam_mask, follicle_mask)
        spatially_consistent = overlap_score >= self.p["iou_agreement_threshold"]

        cnn_says_pcos = cnn_result["predicted_label"] == "infected"
        meets_pcom = follicle_result["follicle_count"] >= self.pcom_threshold
        diagnostically_consistent = cnn_says_pcos == meets_pcom

        agree = spatially_consistent and diagnostically_consistent

        reasoning = (
            f"CNN predicts '{cnn_result['predicted_label']}' (p={cnn_result['pcos_probability']:.2f}). "
            f"Follicle count is {follicle_result['follicle_count']} "
            f"({'meets' if meets_pcom else 'does not meet'} the >= {self.pcom_threshold} PCOM threshold). "
            f"Grad-CAM/follicle spatial overlap (IoU) is {overlap_score:.2f} "
            f"({'consistent' if spatially_consistent else 'inconsistent'} with the "
            f">= {self.p['iou_agreement_threshold']} threshold)."
        )
        if not diagnostically_consistent:
            reasoning += " Diagnostic mismatch: CNN and follicle count point in different directions."

        return {
            "agent": "grounding",
            "agreement_status": "agree" if agree else "disagree",
            "overlap_score": overlap_score,
            "spatially_consistent": spatially_consistent,
            "diagnostically_consistent": diagnostically_consistent,
            "reasoning": reasoning,
        }
