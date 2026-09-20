"""Turn the raw ODIR-5K Kaggle download into the folder layout the trainer wants.

    python scripts/prepare_odir.py --source ~/Downloads/odir5k

Why this script is not a two-liner
----------------------------------
ODIR-5K labels (the N, D, G, C, A, H, M, O columns) are recorded **per patient**,
not per eye. A patient marked "G" may well have glaucoma in the left eye and a
perfectly healthy right eye. Training on the patient-level labels therefore
teaches the model that healthy retinas are glaucomatous, and the accuracy you
report is measured against wrong ground truth.

The per-eye diagnostic keyword columns are the honest source of labels, so this
script parses those instead. It also:

  * drops images whose keywords mention a quality problem (lens dust, low image
    quality, optic disk not visible) — they are unlabelable, not a disease class;
  * keeps only images that map to exactly one of our five classes, because our
    Model A is single-label; and
  * splits by PATIENT, never by image, so the left and right eye of one person
    can never land in both train and test. Splitting by image leaks information
    and inflates test accuracy by several points.
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import SEED, get_task  # noqa: E402

# Keyword fragments -> our class names. Matched against the lower-cased
# diagnostic keyword string for one eye.
KEYWORD_MAP = {
    "normal": ["normal fundus"],
    "cataract": ["cataract"],
    "glaucoma": ["glaucoma"],
    "amd": ["age-related macular degeneration", "dry age", "wet age"],
    "diabetic_retinopathy": [
        "diabetic retinopathy",
        "proliferative retinopathy",
        "nonproliferative retinopathy",
        "non proliferative retinopathy",
    ],
}

# Keywords that describe a bad photograph rather than a finding.
QUALITY_FLAGS = [
    "lens dust",
    "optic disk photographically invisible",
    "low image quality",
    "image offset",
    "no fundus image",
]


def label_for(keywords: str) -> str | None:
    """Return the single class for one eye, or None if unusable/ambiguous."""
    if not isinstance(keywords, str) or not keywords.strip():
        return None
    text = keywords.lower()

    if any(flag in text for flag in QUALITY_FLAGS):
        return None

    hits = {cls for cls, frags in KEYWORD_MAP.items() if any(f in text for f in frags)}

    # "normal fundus" alongside a disease keyword is contradictory -> drop.
    if hits == {"normal"}:
        return "normal"
    hits.discard("normal")
    return hits.pop() if len(hits) == 1 else None


def find_image(name: str, roots: list[Path], index: dict[str, Path]) -> Path | None:
    if name in index:
        return index[name]
    stem = Path(name).stem.lower()
    return index.get(stem)


def build_index(roots: list[Path]) -> dict[str, Path]:
    """filename and stem -> full path, so we tolerate .jpg/.png differences."""
    index: dict[str, Path] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for p in root.rglob("*"):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                index.setdefault(p.name, p)
                index.setdefault(p.stem.lower(), p)
    return index


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Prepare ODIR-5K for Model A.")
    ap.add_argument("--source", required=True, type=Path,
                    help="Folder containing full_df.csv and the image folders.")
    ap.add_argument("--csv", type=Path, default=None,
                    help="Path to full_df.csv (default: <source>/full_df.csv).")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--max-per-class", type=int, default=0,
                    help="Cap images per class to keep training fast (0 = no cap).")
    ap.add_argument("--move", action="store_true", help="Move files instead of copying.")
    args = ap.parse_args(argv)

    task = get_task("internal")
    csv_path = args.csv or (args.source / "full_df.csv")
    if not csv_path.exists():
        matches = list(args.source.rglob("full_df.csv"))
        if not matches:
            print(f"ERROR: could not find full_df.csv under {args.source}", file=sys.stderr)
            return 1
        csv_path = matches[0]
    print(f"labels : {csv_path}")

    df = pd.read_csv(csv_path)

    left_col = next((c for c in df.columns if "left" in c.lower() and "keyword" in c.lower()), None)
    right_col = next((c for c in df.columns if "right" in c.lower() and "keyword" in c.lower()), None)
    left_img = next((c for c in df.columns if c.lower() in {"left-fundus", "left_fundus"}), None)
    right_img = next((c for c in df.columns if c.lower() in {"right-fundus", "right_fundus"}), None)
    id_col = next((c for c in df.columns if c.strip().lower() == "id"), df.columns[0])

    if not all([left_col, right_col, left_img, right_img]):
        print("ERROR: full_df.csv is missing the per-eye keyword or filename columns.",
              file=sys.stderr)
        print(f"Columns found: {list(df.columns)}", file=sys.stderr)
        return 1

    index = build_index([args.source])
    print(f"images : {len(index) // 2} files indexed under {args.source}")

    # ---- Label every eye ------------------------------------------------
    records = []           # (patient_id, class, source_path)
    dropped = Counter()
    for _, row in df.iterrows():
        pid = str(row[id_col])
        for kw_col, fn_col in ((left_col, left_img), (right_col, right_img)):
            cls = label_for(row[kw_col])
            if cls is None:
                dropped["unusable or ambiguous label"] += 1
                continue
            src = find_image(str(row[fn_col]), [args.source], index)
            if src is None:
                dropped["image file not found"] += 1
                continue
            records.append((pid, cls, src))

    if not records:
        print("ERROR: no usable images. Check --source points at the extracted dataset.",
              file=sys.stderr)
        return 1

    print(f"\nlabelled {len(records)} eyes; dropped {sum(dropped.values())}")
    for reason, n in dropped.items():
        print(f"  - {n:>6} {reason}")
    print("class distribution:", dict(Counter(c for _, c, _ in records)))

    # ---- Optional cap, applied per class --------------------------------
    rng = random.Random(SEED)
    if args.max_per_class > 0:
        by_class = defaultdict(list)
        for r in records:
            by_class[r[1]].append(r)
        capped = []
        for cls, items in by_class.items():
            rng.shuffle(items)
            capped.extend(items[: args.max_per_class])
        records = capped
        print(f"capped to {args.max_per_class}/class -> {len(records)} images")

    # ---- Split by patient, not by image ---------------------------------
    patients = sorted({pid for pid, _, _ in records})
    rng.shuffle(patients)
    n = len(patients)
    n_test = int(n * args.test_frac)
    n_val = int(n * args.val_frac)
    split_of = {}
    for i, pid in enumerate(patients):
        split_of[pid] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")

    # ---- Write ----------------------------------------------------------
    if task.dataset_dir.exists():
        print(f"clearing {task.dataset_dir}")
        shutil.rmtree(task.dataset_dir)

    written = Counter()
    for pid, cls, src in records:
        split = split_of[pid]
        dest_dir = task.dataset_dir / split / cls
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{pid}_{re.sub(r'[^A-Za-z0-9._-]', '_', src.name)}"
        if args.move:
            shutil.move(str(src), dest)
        else:
            shutil.copy2(src, dest)
        written[(split, cls)] += 1

    print(f"\nwrote {sum(written.values())} images to {task.dataset_dir}")
    for split in ("train", "val", "test"):
        row = {c: written[(split, c)] for c in sorted(KEYWORD_MAP) if written[(split, c)]}
        print(f"  {split:<6} {sum(row.values()):>6}  {row}")

    print("\nNext: python -m src.train --task internal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
