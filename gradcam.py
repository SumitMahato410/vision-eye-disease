"""Grad-CAM heatmaps — the explainability layer.

Muchuchuti & Viriri (2023) name explainability as one of the open problems in
retinal-disease deep learning: a model that outputs "glaucoma, 94%" with no
justification is not clinically trustworthy. Grad-CAM answers "which pixels
drove that decision" by weighting the final convolutional feature maps with the
gradient of the predicted class score.

If the heatmap for a diabetic-retinopathy prediction lights up on haemorrhages
near the macula, the model is looking at the right thing. If it lights up on the
black border of the image, the model has learned a dataset artifact — which is a
genuinely useful finding to report.
"""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

# TensorFlow is imported inside compute_heatmap, not here. Importing it at
# module scope costs several seconds and would be paid by the Flask process
# just to render the upload page, which needs no TensorFlow at all.


def compute_heatmap(model, image_batch: np.ndarray, conv_layer: str,
                    class_index: int | None = None) -> np.ndarray:
    """Return a HxW heatmap normalised to [0, 1].

    ``image_batch`` must have shape (1, H, W, 3) in the same range the model was
    trained on (raw 0-255 here — see src/models.py).
    """
    import tensorflow as tf
    from tensorflow import keras

    grad_model = keras.Model(
        model.inputs,
        [model.get_layer(conv_layer).output, model.output],
    )

    x = tf.convert_to_tensor(image_batch, dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_out, predictions = grad_model(x, training=False)
        if class_index is None:
            class_index = int(tf.argmax(predictions[0]))
        score = predictions[:, class_index]

    grads = tape.gradient(score, conv_out)
    if grads is None:  # e.g. a fully frozen graph
        return np.zeros(conv_out.shape[1:3], dtype=np.float32)

    # Channel importance = spatially averaged gradient.
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(conv_out[0] * weights, axis=-1)
    cam = tf.nn.relu(cam).numpy()

    peak = cam.max()
    return cam / peak if peak > 0 else cam


def _colorize(heat: np.ndarray, size: tuple[int, int]) -> Image.Image:
    """Map [0,1] -> an RGBA image using a jet-like ramp (no matplotlib needed)."""
    heat_img = Image.fromarray(np.uint8(heat * 255), mode="L").resize(size, Image.BICUBIC)
    h = np.asarray(heat_img, dtype=np.float32) / 255.0

    # Blue -> cyan -> green -> yellow -> red
    r = np.clip(1.5 - np.abs(4 * h - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * h - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * h - 1), 0, 1)
    # Transparent where the model did not look, so the photo shows through.
    a = np.clip(h * 1.6, 0, 1) * 0.65

    rgba = np.stack([r, g, b, a], axis=-1) * 255
    return Image.fromarray(rgba.astype(np.uint8), mode="RGBA")


def overlay(original: Image.Image, heat: np.ndarray) -> Image.Image:
    base = original.convert("RGBA")
    return Image.alpha_composite(base, _colorize(heat, base.size)).convert("RGB")


def overlay_to_data_uri(original: Image.Image, heat: np.ndarray) -> str:
    """PNG data URI, ready to drop straight into an <img src>."""
    buf = io.BytesIO()
    overlay(original, heat).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
