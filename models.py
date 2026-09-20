"""CNN architectures for both models, built with transfer learning.

Design notes
------------
The backbone is attached with ``input_tensor=`` rather than being called as a
nested model. That keeps the whole network as one flat graph, which is what
makes ``model.get_layer("top_activation")`` work for Grad-CAM later. If you
build it the nested way the conv activations are hidden inside a sub-model and
Grad-CAM becomes painful.

Input convention: **every model in this project accepts raw pixels in [0, 255]**.
Any rescaling a backbone needs is baked in as the first layer, so inference code
never has to remember which normalisation belongs to which backbone.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .config import DROPOUT, TaskConfig

# backbone name -> (constructor, name of final conv activation, needs [-1,1] input)
BACKBONES = {
    "efficientnetb0": (keras.applications.EfficientNetB0, "top_activation", False),
    "efficientnetb3": (keras.applications.EfficientNetB3, "top_activation", False),
    "mobilenetv2": (keras.applications.MobileNetV2, "out_relu", True),
    "resnet50": (keras.applications.ResNet50, "conv5_block3_out", True),
}


def build_augmenter(seed: int = 42) -> keras.Sequential:
    """Augmentation pipeline (objective 5.2 of the synopsis).

    Kept conservative on purpose. Fundus images have a fixed orientation and a
    circular field of view, so heavy shear or vertical flips would create
    photographs that could never come out of a real retinal camera.
    """
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal", seed=seed),
            layers.RandomRotation(0.08, fill_mode="constant", seed=seed),
            layers.RandomZoom(0.12, fill_mode="constant", seed=seed),
            layers.RandomTranslation(0.06, 0.06, fill_mode="constant", seed=seed),
            layers.RandomContrast(0.15, seed=seed),
        ],
        name="augmentation",
    )


def build_model(task: TaskConfig, num_classes: int, dropout: float = DROPOUT):
    """Return (model, backbone_layer_names).

    The second value is the list of layer names belonging to the pretrained
    backbone, which ``unfreeze_top`` uses to fine-tune only the deep layers.
    """
    if task.backbone not in BACKBONES:
        raise ValueError(
            f"Unknown backbone {task.backbone!r}. Choose from {sorted(BACKBONES)}"
        )
    ctor, conv_layer, needs_signed_input = BACKBONES[task.backbone]

    inputs = keras.Input(shape=(task.image_size, task.image_size, 3), name="image")

    # Backbones differ in what they expect. EfficientNet in Keras normalises
    # internally and wants raw [0, 255]; MobileNetV2 and ResNet50 want [-1, 1].
    x = inputs
    if needs_signed_input:
        x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="rescale")(x)

    base = ctor(include_top=False, weights="imagenet", input_tensor=x)
    base.trainable = False

    y = layers.GlobalAveragePooling2D(name="gap")(base.output)
    y = layers.Dropout(dropout, name="head_dropout")(y)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(y)

    model = keras.Model(inputs, outputs, name=f"vision_{task.key}")

    # Sanity check: the layer Grad-CAM will look for must actually exist.
    model.get_layer(conv_layer)

    return model, [l.name for l in base.layers]


def unfreeze_top(model, backbone_layers: list[str], n_layers: int) -> int:
    """Unfreeze the last ``n_layers`` of the backbone for stage-2 fine-tuning.

    BatchNormalization layers stay frozen. Un-freezing them with a small batch
    size makes the running statistics drift and the validation accuracy
    collapse, which is the single most common transfer-learning bug.
    """
    to_unfreeze = set(backbone_layers[-n_layers:]) if n_layers > 0 else set()
    count = 0
    for layer in model.layers:
        if layer.name in to_unfreeze and not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
            count += 1
    return count


def compile_model(model, lr: float, label_smoothing: float = 0.0):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc", multi_label=False),
        ],
    )
    return model


def conv_layer_name(task: TaskConfig) -> str:
    """Name of the last conv activation, used by Grad-CAM."""
    if task.gradcam_layer:
        return task.gradcam_layer
    return BACKBONES[task.backbone][1]
