"""VISION — Flask web application.

    python app.py                      # development
    gunicorn -w 2 -t 120 app:app       # production

Endpoints
    GET  /                 the single-page UI
    GET  /api/health       which models are trained and loadable
    GET  /api/tasks        task metadata and class lists
    POST /api/predict      multipart form: image=<file>, task=internal|external
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from src import config as cfg
from src import inference

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("vision")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = cfg.MAX_UPLOAD_MB * 1024 * 1024
app.config["JSON_SORT_KEYS"] = False


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------


@app.get("/")
def index():
    return render_template(
        "index.html",
        tasks=[
            {
                "key": t.key,
                "name": t.name,
                "blurb": t.blurb,
                "available": inference.model_available(t.key),
            }
            for t in cfg.TASKS.values()
        ],
        max_mb=cfg.MAX_UPLOAD_MB,
    )


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------


@app.get("/api/health")
def health():
    models = {}
    for key, task in cfg.TASKS.items():
        available = inference.model_available(key)
        entry = {"available": available, "path": str(task.model_path)}
        if available:
            entry["size_mb"] = round(task.model_path.stat().st_size / 1e6, 1)
            try:
                entry["classes"] = json.loads(task.classes_path.read_text())
            except Exception:
                entry["classes"] = None
        models[key] = entry
    return jsonify({"status": "ok", "models": models})


@app.get("/api/tasks")
def tasks():
    out = []
    for task in cfg.TASKS.values():
        item = {
            "key": task.key,
            "name": task.name,
            "blurb": task.blurb,
            "backbone": task.backbone,
            "image_size": task.image_size,
            "available": inference.model_available(task.key),
            "classes": task.default_classes,
        }
        metrics_file = cfg.REPORT_DIR / f"{task.key}_metrics.json"
        if metrics_file.exists():
            try:
                m = json.loads(metrics_file.read_text())
                item["test_metrics"] = {
                    "accuracy": m.get("accuracy"),
                    "macro_f1": m.get("macro_f1"),
                    "macro_auc_ovr": m.get("macro_auc_ovr"),
                    "n_images": m.get("n_images"),
                }
            except Exception:
                pass
        out.append(item)
    return jsonify({"tasks": out})


@app.post("/api/predict")
def api_predict():
    task_key = (request.form.get("task") or "").strip().lower()
    if task_key not in cfg.TASKS:
        return jsonify({"error": f"Unknown task. Use one of: {sorted(cfg.TASKS)}"}), 400

    file = request.files.get("image")
    if file is None or not file.filename:
        return jsonify({"error": "No image uploaded. Send a file in the 'image' field."}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in cfg.ALLOWED_EXTENSIONS:
        return jsonify({
            "error": f"Unsupported file type '{ext}'. "
                     f"Allowed: {', '.join(sorted(cfg.ALLOWED_EXTENSIONS))}"
        }), 400

    data = file.read()
    if not data:
        return jsonify({"error": "Uploaded file is empty."}), 400

    want_cam = request.form.get("gradcam", "true").lower() != "false"

    try:
        result = inference.predict(task_key, data, want_gradcam=want_cam)
    except FileNotFoundError:
        return jsonify({
            "error": f"The '{task_key}' model has not been trained yet.",
            "hint": f"Run: python -m src.train --task {task_key}",
        }), 503
    except OSError:
        return jsonify({"error": "That file could not be read as an image."}), 400
    except Exception as exc:                      # noqa: BLE001
        log.exception("prediction failed")
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    return jsonify(result)


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": f"File too large. Maximum is {cfg.MAX_UPLOAD_MB} MB."}), 413


@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "Not found"}), 404


# --------------------------------------------------------------------------

if __name__ == "__main__":
    for key in cfg.TASKS:
        state = "ready" if inference.model_available(key) else "NOT TRAINED"
        log.info("model '%s': %s", key, state)

    if os.getenv("VISION_WARMUP", "").lower() in {"1", "true", "yes"}:
        for key in cfg.TASKS:
            log.info("warm-up '%s': %s", key, inference.warm_up(key))

    app.run(
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "").lower() in {"1", "true", "yes"},
    )
