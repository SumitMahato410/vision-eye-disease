"""Train either model.

    python -m src.train --task internal
    python -m src.train --task external --head-epochs 5 --finetune-epochs 10

Training runs in two stages, which is the standard recipe for small medical
datasets:

  Stage 1  backbone frozen, only the new classifier head learns. Fast, and it
           stops the large random gradients from the untrained head from
           destroying the pretrained ImageNet filters.
  Stage 2  the top ~40 backbone layers are unfrozen and everything is retrained
           at a very low learning rate so the deep, task-specific filters adapt
           to retinal texture without forgetting the generic edge detectors.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time

import numpy as np
from tensorflow import keras

from . import config as cfg
from .data import (
    compute_class_weights,
    dataset_summary,
    discover_classes,
    load_split,
    save_classes,
)
from .models import build_model, compile_model, unfreeze_top


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train a VISION eye-disease model.")
    p.add_argument("--task", required=True, choices=sorted(cfg.TASKS))
    p.add_argument("--head-epochs", type=int, default=cfg.HEAD_EPOCHS)
    p.add_argument("--finetune-epochs", type=int, default=cfg.FINETUNE_EPOCHS)
    p.add_argument("--head-lr", type=float, default=cfg.HEAD_LR)
    p.add_argument("--finetune-lr", type=float, default=cfg.FINETUNE_LR)
    p.add_argument("--unfreeze", type=int, default=cfg.UNFREEZE_LAYERS)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--no-class-weights", action="store_true")
    return p.parse_args(argv)


def make_callbacks(task, monitor="val_auc", mode="max"):
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(task.model_path),
            monitor=monitor,
            mode=mode,
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode=mode,
            patience=cfg.EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor, mode=mode, factor=0.3, patience=2, min_lr=1e-7, verbose=1
        ),
    ]


def main(argv=None) -> int:
    args = parse_args(argv)
    task = cfg.get_task(args.task)
    if args.batch_size:
        task = dataclasses.replace(task, batch_size=args.batch_size)

    classes = discover_classes(task)
    summary = dataset_summary(task, classes)
    print(json.dumps(summary, indent=2))

    train_ds = load_split(task, "train", classes, shuffle=True, augment=True)
    val_ds = load_split(task, "val", classes, shuffle=False)

    class_weight = None if args.no_class_weights else compute_class_weights(task, classes)
    if class_weight:
        print("class weights:", {classes[i]: round(w, 3) for i, w in class_weight.items()})

    model, backbone_layers = build_model(task, num_classes=len(classes))
    compile_model(model, args.head_lr, cfg.LABEL_SMOOTHING)
    trainable = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    print(f"{task.name}: {model.count_params():,} parameters ({trainable:,} trainable)")

    started = time.time()

    # ---- Stage 1 -------------------------------------------------------
    print("\n=== Stage 1: training classifier head (backbone frozen) ===")
    h1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.head_epochs,
        class_weight=class_weight,
        callbacks=make_callbacks(task),
    )

    # ---- Stage 2 -------------------------------------------------------
    history = {k: list(map(float, v)) for k, v in h1.history.items()}
    if args.finetune_epochs > 0 and args.unfreeze > 0:
        n = unfreeze_top(model, backbone_layers, args.unfreeze)
        print(f"\n=== Stage 2: fine-tuning {n} backbone layers at lr={args.finetune_lr} ===")
        compile_model(model, args.finetune_lr, cfg.LABEL_SMOOTHING)
        h2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.head_epochs + args.finetune_epochs,
            initial_epoch=len(h1.history.get("loss", [])),
            class_weight=class_weight,
            callbacks=make_callbacks(task),
        )
        for k, v in h2.history.items():
            history.setdefault(k, []).extend(map(float, v))

    save_classes(task, classes)
    model.save(task.model_path)

    task.history_path.write_text(
        json.dumps(
            {
                "task": task.key,
                "classes": classes,
                "dataset": summary["counts"],
                "minutes": round((time.time() - started) / 60, 2),
                "history": history,
            },
            indent=2,
        )
    )

    print(f"\nSaved model  -> {task.model_path}")
    print(f"Saved classes-> {task.classes_path}")
    print(f"Saved history-> {task.history_path}")
    print(f"\nNext: python -m src.evaluate --task {task.key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
