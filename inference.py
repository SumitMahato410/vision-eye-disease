"""Prediction engine used by the web app.

Models are loaded lazily and cached, so starting the server is instant and a
missing model only affects the task that needs it — the other one keeps working.
That matters on free hosting tiers where the whole container is restarted often.
"""

from __future__ import annotations

import io
import threading

import numpy as np
from PIL import Image, ImageOps

from . import advice as advice_mod
from . import config as cfg
from .gradcam import compute_heatmap, overlay_to_data_uri

_lock = threading.Lock()
_cache: dict[str, tuple] = {}   # task_key -> (model, classes)


def _load(task_key: str):
    """Load and cache (model, classes) for a task. Thread-safe."""
    if task_key in _cache:
        return _cache[task_key]
    with _lock:
        if task_key in _cache:          # another thread won the race
            return _cache[task_key]

        # Check the cheap thing first. Importing TensorFlow costs seconds, so
        # an untrained model should fail before we pay that, not after.
        task = cfg.get_task(task_key)
        if not task.model_path.exists():
            raise FileNotFoundError(
                f"Model for task '{task_key}' not found at {task.model_path}."
            )

        # Imported here, not at module scope: TensorFlow takes several seconds
        # to import and we do not want to pay that cost just to serve the page.
        try:
            from tensorflow import keras
        except ImportError as exc:      # pragma: no cover
            raise RuntimeError(
                "TensorFlow is not installed. Run: pip install -r requirements.txt"
            ) from exc
        from .data import load_classes

        model = keras.models.load_model(task.model_path)
        classes = load_classes(task)
        _cache[task_key] = (model, classes)
        return _cache[task_key]


def model_available(task_key: str) -> bool:
    task = cfg.get_task(task_key)
    return task.model_path.exists() and task.classes_path.exists()


def warm_up(task_key: str) -> bool:
    """Pre-load a model so the first real request is not slow."""
    try:
        _load(task_key)
        return True
    except Exception:
        return False


def open_image(data: bytes) -> Image.Image:
    """Decode upload bytes into an RGB PIL image, honouring EXIF rotation."""
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)      # phone photos are often rotated
    return img.convert("RGB")


def preprocess(img: Image.Image, size: int) -> np.ndarray:
    """Resize to the model input and return a (1, size, size, 3) float array.

    Values stay in [0, 255] — every model in this project does its own
    normalisation internally (see src/models.py).
    """
    resized = img.resize((size, size), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


def predict(task_key: str, image_bytes: bytes, want_gradcam: bool = True) -> dict:
    """Run one image through one model and return a JSON-serialisable result."""
    task = cfg.get_task(task_key)
    model, classes = _load(task_key)

    img = open_image(image_bytes)
    batch = preprocess(img, task.image_size)

    probs = model.predict(batch, verbose=0)[0].astype(float)
    order = np.argsort(probs)[::-1]
    top_idx = int(order[0])
    top_prob = float(probs[top_idx])
    top_class = classes[top_idx]

    low_confidence = top_prob < cfg.LOW_CONFIDENCE_THRESHOLD
    info = advice_mod.get(top_class)

    result = {
        "task": task.key,
        "task_name": task.name,
        "prediction": top_class,
        "label": info["label"],
        "confidence": round(top_prob, 4),
        "low_confidence": low_confidence,
        "urgency": "routine" if low_confidence else info["urgency"],
        "summary": info["summary"],
        "prevention": info["prevention"],
        "next_steps": info["next_steps"],
        "disclaimer": advice_mod.DISCLAIMER,
        "probabilities": [
            {
                "class": classes[i],
                "label": advice_mod.pretty(classes[i]),
                "probability": round(float(probs[i]), 4),
            }
            for i in order
        ],
    }

    if low_confidence:
        result["message"] = (
            "The model is not confident about this image. That usually means the photo "
            "is blurred, badly lit, cropped too far out, or is not the kind of photo "
            "this model expects. Try again with a sharper, well-lit, close-up image."
        )

    if want_gradcam:
        try:
            heat = compute_heatmap(model, batch, task.gradcam_layer, top_idx)
            result["gradcam"] = overlay_to_data_uri(img, heat)
        except Exception as exc:          # explainability is a bonus, never fatal
            result["gradcam_error"] = str(exc)

    return result
