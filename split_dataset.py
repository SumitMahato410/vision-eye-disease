"""Split a flat class-folder dataset into train/val/test.

Use this for Model B (the external eye dataset), and for any Kaggle dataset that
arrives as one folder per class with no split::

    downloaded/
      conjunctivitis/*.jpg
      normal/*.jpg
      stye/*.jpg

    python scripts/split_dataset.py --source downloaded --task external

It also flags near-duplicate filenames and unreadable images, both of which are
common in scraped eye-photo datasets and both of which quietly corrupt your test
accuracy if left in.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import SEED, TASKS, get_task  # noqa: E402

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def file_hash(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Split a class-folder dataset.")
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--min-size", type=int, default=64,
                    help="Reject images smaller than this on either side.")
    ap.add_argument("--keep-duplicates", action="store_true")
    args = ap.parse_args(argv)

    task = get_task(args.task)
    if not args.source.is_dir():
        print(f"ERROR: {args.source} is not a directory", file=sys.stderr)
        return 1

    class_dirs = sorted(p for p in args.source.iterdir() if p.is_dir())
    if len(class_dirs) < 2:
        print(f"ERROR: expected at least 2 class folders inside {args.source}, "
              f"found {len(class_dirs)}", file=sys.stderr)
        return 1

    rng = random.Random(SEED)
    seen_hashes: set[str] = set()
    rejected = Counter()
    plan: list[tuple[str, str, Path]] = []   # (split, class, path)

    for cdir in class_dirs:
        files = [p for p in sorted(cdir.rglob("*")) if p.suffix.lower() in IMAGE_EXT]
        good = []
        for p in files:
            try:
                with Image.open(p) as im:
                    im.verify()
                with Image.open(p) as im:
                    w, h = im.size
            except Exception:
                rejected["corrupt or unreadable"] += 1
                continue
            if min(w, h) < args.min_size:
                rejected["too small"] += 1
                continue
            if not args.keep_duplicates:
                digest = file_hash(p)
                if digest in seen_hashes:
                    rejected["exact duplicate"] += 1
                    continue
                seen_hashes.add(digest)
            good.append(p)

        rng.shuffle(good)
        n = len(good)
        n_test = int(n * args.test_frac)
        n_val = int(n * args.val_frac)
        for i, p in enumerate(good):
            split = "test" if i < n_test else ("val" if i < n_test + n_val else "train")
            plan.append((split, cdir.name, p))
        print(f"{cdir.name:<28} {n:>6} usable")

    if rejected:
        print("\nrejected:", dict(rejected))

    if task.dataset_dir.exists():
        print(f"clearing {task.dataset_dir}")
        shutil.rmtree(task.dataset_dir)

    written = Counter()
    for split, cls, src in plan:
        dest_dir = task.dataset_dir / split / cls
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest_dir / src.name)
        written[(split, cls)] += 1

    print(f"\nwrote {sum(written.values())} images to {task.dataset_dir}")
    for split in ("train", "val", "test"):
        row = {c.name: written[(split, c.name)] for c in class_dirs if written[(split, c.name)]}
        print(f"  {split:<6} {sum(row.values()):>6}  {row}")

    smallest = min((written[(s, c.name)] for s in ("train",) for c in class_dirs), default=0)
    if smallest < 50:
        print(f"\nWarning: the smallest training class has only {smallest} images. "
              f"Expect unstable results; consider collecting more or merging classes.")

    print(f"\nNext: python -m src.train --task {task.key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
