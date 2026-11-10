"""
Grad-CAM helpers — used by the CNN Classification Agent to explain its
prediction, and later by the Grounding Agent to compare that explanation
against the Follicle Counting Agent's detections.
"""

import cv2
import numpy as np
import tensorflow as tf

def heatmap_to_mask(heatmap, target_shape, threshold=0.5):
    """Resizes the Grad-CAM heatmap to target_shape and binarizes it,
    so it can be compared against the follicle mask via IoU."""
    resized = cv2.resize(heatmap, (target_shape[1], target_shape[0]))
    return (resized >= threshold).astype(np.uint8)
    
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="backbone"):
    """Returns a 2D float array in [0, 1]."""
    base_model = model.get_layer(last_conv_layer_name)

    # Model #1: input -> backbone's feature map output
    last_conv_layer_model = tf.keras.Model(base_model.input, base_model.output)

    # Model #2: feature map -> everything after the backbone (GAP, dropout, dense...)
    classifier_input = tf.keras.Input(shape=base_model.output.shape[1:])
    x = classifier_input
    base_index = model.layers.index(base_model)
    for layer in model.layers[base_index + 1:]:
        x = layer(x)
    classifier_model = tf.keras.Model(classifier_input, x)

    with tf.GradientTape() as tape:
        conv_outputs = last_conv_layer_model(img_array)
        tape.watch(conv_outputs)
        predictions = classifier_model(conv_outputs)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_heatmap(img_bgr, heatmap, alpha=0.4):
    heatmap_resized = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    return cv2.addWeighted(img_bgr, 1 - alpha, heatmap_color, alpha, 0)
