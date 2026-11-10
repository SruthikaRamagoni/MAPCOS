"""
CNN Classification Agent
--------------------------
Fine-tuned EfficientNetB0. Outputs PCOS probability + Grad-CAM heatmap.
"""

import cv2
import numpy as np

from mapcos.config import IMG_SIZE, LAST_CONV_LAYER
from mapcos.utils.gradcam import make_gradcam_heatmap, overlay_heatmap


class CNNClassificationAgent:
    def __init__(self, model, last_conv_layer: str = LAST_CONV_LAYER, img_size: int = IMG_SIZE):
        self.model = model
        self.last_conv_layer = last_conv_layer
        self.img_size = img_size

    def _preprocess(self, image_path: str):
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.img_size, self.img_size))
        img_array = np.expand_dims(img_resized / 255.0, axis=0)
        return img_bgr, img_array

    def run(self, image_path: str) -> dict:
        img_bgr, img_array = self._preprocess(image_path)
        prob = float(self.model.predict(img_array, verbose=0)[0][0])
        heatmap = make_gradcam_heatmap(img_array, self.model, self.last_conv_layer)
        resized_bgr = cv2.resize(img_bgr, (self.img_size, self.img_size))
        overlay = overlay_heatmap(resized_bgr, heatmap)

        return {
            "agent": "cnn_classification",
            "pcos_probability": prob,
            "predicted_label": "infected" if prob >= 0.5 else "notinfected",
            "image_shape": resized_bgr.shape[:2],
            "gradcam_heatmap": heatmap,
            "gradcam_overlay": overlay,
        }
