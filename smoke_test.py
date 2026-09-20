"""Prove the whole pipeline runs, before you have downloaded a single dataset.

    python scripts/smoke_test.py

It generates a small synthetic dataset of coloured, textured discs, runs a real
one-epoch training on it, evaluates, then calls the inference path and the Flask
API exactly as the browser does. If this passes, every moving part — data
loading, augmentation, the model graph, checkpointing, metrics, Grad-CAM, the
HTTP layer — is wired up correctly, and any later failure is about your data
rather than your code.

The synthetic classes are visually separable on purpose, so a passing run should
reach high accuracy in one epoch. That is a plumbing check, not a result: do not
put these numbers in your report.

Use --keep to leave the synthetic models in place, or omit it (default) to
restore whatever was in models/ before.
"""

from __future__ import annotations

import argparse
import io
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as cfg  # noqa: E402

RNG = np.random.default_rng(0)

# Three synthetic "conditions": a plain disc, a disc with dark spots (standing
# in for haemorrhages), and a disc with a bright central ring (standing in for a
# cupped optic disc).
SYNTH_CLASSES = ["synth_normal", "synth_spots", "synth_ring"]


def make_image(cls: str, size: int = 256) -> Image.Image:
    img = Image.new("RGB", (size, size), (12, 8, 6))
    d = ImageDraw.Draw(img)
    cx = cy = size // 2
    r = int(size * 0.45)

    base = (170 + RNG.integers(-20, 20), 80 + RNG.integers(-15, 15), 40 + RNG.integers(-10, 10))
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=tuple(int(v) for v in base))

    if cls == "synth_spots":
        for _ in range(RNG.integers(8, 16)):
            a = RNG.uniform(0, 2 * np.pi)
            dist = RNG.uniform(0, r * 0.8)
            x, y = cx + dist * np.cos(a), cy + dist * np.sin(a)
            s = RNG.integers(4, 10)
            d.ellipse([x - s, y - s, x + s, y + s], fill=(70, 20, 20))
    elif cls == "synth_ring":
        rr = int(r * 0.35)
        d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=(240, 225, 180), width=8)

    img = img.filter(ImageFilter.GaussianBlur(RNG.uniform(0.3, 1.2)))
    noise = RNG.normal(0, 7, (size, size, 3))
    arr = np.clip(np.asarray(img, dtype=float) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def build_dataset(root: Path, per_class: dict[str, int]) -> None:
    for split, n in per_class.items():
        for cls in SYNTH_CLASSES:
            d = root / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                make_image(cls).save(d / f"{cls}_{i:03d}.jpg", quality=88)


def step(msg: str) -> None:
    print(f"\n{'=' * 64}\n{msg}\n{'=' * 64}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="End-to-end pipeline smoke test.")
    ap.add_argument("--task", default="external", choices=sorted(cfg.TASKS))
    ap.add_argument("--per-class", type=int, default=24)
    ap.add_argument("--keep", action="store_true",
                    help="Keep the synthetic model instead of restoring the old one.")
    args = ap.parse_args(argv)

    task = cfg.get_task(args.task)
    backup = None
    tmp_data = Path(tempfile.mkdtemp(prefix="vision_smoke_"))

    # Preserve anything real that is already on disk.
    if task.dataset_dir.exists():
        backup = tmp_data / "real_data_backup"
        shutil.move(str(task.dataset_dir), str(backup))
    model_backup = None
    if task.model_path.exists() and not args.keep:
        model_backup = tmp_data / task.model_path.name
        shutil.copy2(task.model_path, model_backup)

    try:
        step(f"1/5  Generating synthetic dataset in {task.dataset_dir}")
        build_dataset(
            task.dataset_dir,
            {"train": args.per_class, "val": max(4, args.per_class // 3),
             "test": max(4, args.per_class // 3)},
        )
        print(f"classes: {SYNTH_CLASSES}")

        step("2/5  Training (1 epoch head only — plumbing check, not a result)")
        from src.train import main as train_main
        rc = train_main([
            "--task", task.key,
            "--head-epochs", "1",
            "--finetune-epochs", "0",
            "--batch-size", "8",
        ])
        assert rc == 0, "training returned non-zero"
        assert task.model_path.exists(), "no model file was written"
        assert task.classes_path.exists(), "no class list was written"

        step("3/5  Evaluating on the synthetic test split")
        from src.evaluate import evaluate
        metrics = evaluate(task.key, "test")
        assert 0.0 <= metrics["accuracy"] <= 1.0

        step("4/5  Inference + Grad-CAM through src.inference")
        from src import inference
        buf = io.BytesIO()
        make_image("synth_spots").save(buf, format="JPEG")
        result = inference.predict(task.key, buf.getvalue(), want_gradcam=True)
        assert result["prediction"] in SYNTH_CLASSES
        assert 0.0 <= result["confidence"] <= 1.0
        assert abs(sum(p["probability"] for p in result["probabilities"]) - 1.0) < 0.05
        if "gradcam" in result:
            print(f"Grad-CAM produced ({len(result['gradcam']) // 1024} KB data URI)")
        else:
            print(f"Grad-CAM unavailable: {result.get('gradcam_error')}")
        print(f"prediction: {result['prediction']} @ {result['confidence']:.3f}")

        step("5/5  HTTP layer via the Flask test client")
        import app as flask_app
        client = flask_app.app.test_client()

        health = client.get("/api/health")
        assert health.status_code == 200, health.status_code
        assert health.get_json()["models"][task.key]["available"] is True

        assert client.get("/").status_code == 200
        assert client.get("/api/tasks").status_code == 200

        buf.seek(0)
        resp = client.post(
            "/api/predict",
            data={"image": (buf, "eye.jpg"), "task": task.key, "gradcam": "false"},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200, resp.get_json()
        assert "prediction" in resp.get_json()

        # Error paths should fail cleanly, not with a 500.
        assert client.post("/api/predict", data={"task": task.key},
                           content_type="multipart/form-data").status_code == 400
        assert client.post("/api/predict",
                           data={"image": (io.BytesIO(b"x"), "a.jpg"), "task": "nope"},
                           content_type="multipart/form-data").status_code == 400

        print("\n" + "=" * 64)
        print("PASS — every stage of the pipeline works end to end.")
        print("Now download the real datasets and see README section 4.")
        print("=" * 64)
        return 0

    finally:
        shutil.rmtree(task.dataset_dir, ignore_errors=True)
        if backup and backup.exists():
            shutil.move(str(backup), str(task.dataset_dir))
        if model_backup and model_backup.exists():
            shutil.copy2(model_backup, task.model_path)
        elif not args.keep:
            task.model_path.unlink(missing_ok=True)
            task.classes_path.unlink(missing_ok=True)
        shutil.rmtree(tmp_data, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
