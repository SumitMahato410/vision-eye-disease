"""Central configuration for the VISION eye-disease screening system.

Everything that the training scripts, the evaluation scripts and the Flask app
need to agree on lives here, so there is exactly one place to change a path,
an image size or a class list.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.getenv("VISION_DATA_DIR", ROOT / "data"))
MODEL_DIR = Path(os.getenv("VISION_MODEL_DIR", ROOT / "models"))
REPORT_DIR = Path(os.getenv("VISION_REPORT_DIR", ROOT / "reports"))

for _d in (DATA_DIR, MODEL_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Task definitions
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskConfig:
    """One of the two models in the system."""

    key: str                  # short id used in URLs and filenames
    name: str                 # human readable name shown in the UI
    blurb: str                # one line describing what kind of photo it wants
    backbone: str             # "efficientnetb0" or "mobilenetv2"
    image_size: int           # square input size in pixels
    batch_size: int
    default_classes: list = field(default_factory=list)
    # Layer whose activations Grad-CAM reads. Set by models.py per backbone.
    gradcam_layer: str = ""

    @property
    def dataset_dir(self) -> Path:
        """Where the split dataset lives: <data>/<key>/{train,val,test}/<class>/"""
        return DATA_DIR / self.key

    @property
    def model_path(self) -> Path:
        return MODEL_DIR / f"{self.key}.keras"

    @property
    def classes_path(self) -> Path:
        return MODEL_DIR / f"{self.key}_classes.json"

    @property
    def history_path(self) -> Path:
        return REPORT_DIR / f"{self.key}_history.json"


INTERNAL = TaskConfig(
    key="internal",
    name="Model A — Internal (retina / fundus)",
    blurb="A fundus photograph taken with a retinal camera.",
    backbone="efficientnetb0",
    image_size=224,
    batch_size=32,
    default_classes=["amd", "cataract", "diabetic_retinopathy", "glaucoma", "normal"],
    gradcam_layer="top_activation",
)

EXTERNAL = TaskConfig(
    key="external",
    name="Model B — External (close-up eye photo)",
    blurb="A normal close-up photo of the eye, e.g. from a phone camera.",
    backbone="mobilenetv2",
    image_size=224,
    batch_size=32,
    default_classes=["conjunctivitis", "normal", "stye"],
    gradcam_layer="out_relu",
)

TASKS = {t.key: t for t in (INTERNAL, EXTERNAL)}


def get_task(key: str) -> TaskConfig:
    if key not in TASKS:
        raise KeyError(f"Unknown task {key!r}. Valid keys: {sorted(TASKS)}")
    return TASKS[key]


# --------------------------------------------------------------------------
# Training defaults (overridable from the CLI)
# --------------------------------------------------------------------------

SEED = 42
HEAD_EPOCHS = 8            # stage 1: frozen backbone, train the new head only
FINETUNE_EPOCHS = 12       # stage 2: unfreeze the top of the backbone
HEAD_LR = 1e-3
FINETUNE_LR = 1e-5
UNFREEZE_LAYERS = 40       # how many trailing backbone layers to unfreeze
DROPOUT = 0.3
LABEL_SMOOTHING = 0.05
EARLY_STOP_PATIENCE = 5


# --------------------------------------------------------------------------
# Web app
# --------------------------------------------------------------------------

MAX_UPLOAD_MB = 8
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Below this top-1 probability the app refuses to name a condition and tells
# the user the photo was not clear enough. Screening tools should abstain.
LOW_CONFIDENCE_THRESHOLD = 0.45
