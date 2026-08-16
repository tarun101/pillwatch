#!/usr/bin/env python3
"""Baseline pill-presence classifier for cropped pillbox cells.

Works on the aligned cell crops produced by detect/crop_cells.py. Pills seen
through the tinted lids show up as compact local blobs, while the things that
fooled naive frame differencing — auto-exposure drift, white-balance shifts,
moving specular highlights on the glossy lids — are smooth, low-frequency
changes. So each cell is scored by its band-pass (difference-of-Gaussians)
blob energy:

  1. compute |DoG| per Lab channel over the cell interior,
  2. zero out the printed day/slot text (much darker than the lid),
  3. subtract the same cell's response in a reference photo of the EMPTY box
     (dilated, to tolerate a few px of residual misalignment) — this cancels
     cell walls, dividers and any leftover text halo,
  4. score = fraction of pixels with significant residual energy.

A cell is called "pill" when the score exceeds AREA_THRESHOLD.

Usage:
    python3 detect/classify_cells.py [--cells DIR] [--out results.json]
                                     [--annotate DIR]

Outputs JSON mapping photo -> {DAY_SLOT: {"pill": bool, "score": float}},
and with --annotate writes per-photo grid images with green (empty) / red
(pill) cell borders and scores, for visual QA.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

try:  # works both as `-m detect.classify_cells` and as a bare script
    from . import crop_cells
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from detect import crop_cells

# The empty-box reference photo and grid layout follow the active camera
# profile (see detect/camera_profiles.py). For imx708 these are the original
# values, so nothing changes on the Pi.
REF_STEM = Path(crop_cells.REF_IMAGE).stem  # photo of the completely empty box
REF_DIR = Path(__file__).parent / "reference_cells"  # empty-box cell crops

DAYS = list(crop_cells.DAYS)
SLOTS = list(crop_cells.SLOTS)

# Ignore a border strip of each crop: cell walls, divider bleed and the
# strongest specular highlights live there, pills sit in the middle.
MARGIN = 0.16
DOG_SIGMA_FINE = 5    # pill-sized blob scale (crops are 280x380)
DOG_SIGMA_COARSE = 21
TEXT_DARKNESS = 25    # L* below cell median that counts as printed text
REF_DILATE = 15       # px of tolerance when cancelling reference structure
REF_GAIN = 1.3        # overshoot so reference structure fully cancels
PIXEL_THRESHOLD = 12  # residual DoG energy that counts as "blob"
AREA_THRESHOLD = 0.00719  # fraction of cell area with blobs -> pill


def load_cell(path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    h, w = img.shape[:2]
    my, mx = int(h * MARGIN), int(w * MARGIN)
    return img[my:h - my, mx:w - mx]


def dog_response(bgr):
    """Per-pixel band-pass blob energy, with the printed text zeroed out."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]
    text = (L < np.median(L) - TEXT_DARKNESS).astype(np.uint8)
    text = cv2.dilate(text, np.ones((9, 9), np.uint8))
    dog = np.zeros(L.shape, np.float32)
    for ch in range(3):
        c = lab[:, :, ch]
        band = (cv2.GaussianBlur(c, (0, 0), DOG_SIGMA_FINE)
                - cv2.GaussianBlur(c, (0, 0), DOG_SIGMA_COARSE))
        dog += np.abs(band)
    dog[text > 0] = 0
    return dog


def cell_score(cell, ref_dog):
    """Fraction of the cell interior with blob energy not explained by the
    empty reference."""
    dog = dog_response(cell)
    ref = cv2.dilate(ref_dog, np.ones((REF_DILATE, REF_DILATE), np.uint8))
    resid = np.maximum(0, dog - REF_GAIN * ref)
    return float((resid > PIXEL_THRESHOLD).mean())


class AnalysisError(Exception):
    """Photo could not be analysed; str(exc) explains why."""


_ref_dog = None  # lazy: per-cell DoG response of the empty-box reference crops


def _crop_margin(img):
    """Drop the MARGIN border strip, matching load_cell() for whole crops."""
    h, w = img.shape[:2]
    my, mx = int(h * MARGIN), int(w * MARGIN)
    return img[my:h - my, mx:w - mx]


def _load_refs():
    global _ref_dog
    if _ref_dog is not None:
        return _ref_dog
    refs = {}
    for day in DAYS:
        for slot in SLOTS:
            key = f"{day}_{slot}"
            img = cv2.imread(str(REF_DIR / f"{key}.jpg"))
            if img is None:
                raise AnalysisError(f"missing reference crop {REF_DIR / f'{key}.jpg'}")
            refs[key] = dog_response(_crop_margin(img))
    _ref_dog = refs
    return refs


def analyze(photo_path):
    """Analyse one photo with the difference-of-Gaussians blob detector.

    Locates and warps the box (same front-end as the CNN pipeline), then
    scores each cell's band-pass blob energy against the empty-box reference.
    Returns {DAY_SLOT: {"pill": bool, "score": float}}.
    """
    refs = _load_refs()
    img = cv2.imread(str(photo_path))
    if img is None:
        raise AnalysisError(f"cannot read image {photo_path}")
    ref_photo = Path(__file__).parent.parent / "images" / crop_cells.REF_IMAGE
    if not ref_photo.is_file():
        raise AnalysisError(f"reference photo not found at {ref_photo}")
    templates = crop_cells.build_matcher(ref_photo)
    quad, confs = crop_cells.align_quad(img, templates)
    if quad is None:
        conf_str = "/".join(f"{c:.2f}" for c in confs)
        raise AnalysisError(
            f"pillbox not found in photo (anchor confidence {conf_str}) — "
            "is the box in its usual spot?")
    grid = crop_cells.warp_grid(img, quad)
    out = {}
    for day, slot, crop in crop_cells.cell_crops(grid):
        key = f"{day}_{slot}"
        score = cell_score(_crop_margin(crop), refs[key])
        out[key] = {"pill": bool(score > AREA_THRESHOLD), "score": round(score, 4)}
    return out


def annotate(cells_dir, results, out_path):
    rows = []
    for slot in SLOTS:
        row = []
        for day in DAYS:
            img = cv2.imread(str(cells_dir / f"{day}_{slot}.jpg"))
            r = results[f"{day}_{slot}"]
            color = (0, 0, 255) if r["pill"] else (0, 200, 0)
            cv2.rectangle(img, (4, 4), (img.shape[1] - 5, img.shape[0] - 5),
                          color, 8)
            cv2.putText(img, f"{r['score']:.3f}", (14, img.shape[0] - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            row.append(img)
        rows.append(np.hstack(row))
    cv2.imwrite(str(out_path), cv2.resize(np.vstack(rows), None, fx=0.5, fy=0.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cells", default="dataset/cells",
                    help="directory produced by crop_cells.py")
    ap.add_argument("--out", default="dataset/results.json")
    ap.add_argument("--annotate", metavar="DIR",
                    help="also write annotated QA grids to DIR")
    args = ap.parse_args()

    cells_root = Path(args.cells)
    ref_dir = cells_root / REF_STEM
    if not ref_dir.is_dir():
        sys.exit(f"error: reference crops not found at {ref_dir} "
                 "(run crop_cells.py first)")
    ref = {}
    for day in DAYS:
        for slot in SLOTS:
            ref[f"{day}_{slot}"] = dog_response(
                load_cell(ref_dir / f"{day}_{slot}.jpg"))

    all_results = {}
    for photo_dir in sorted(p for p in cells_root.iterdir()
                            if p.is_dir() and not p.name.startswith(".")):
        results = {}
        for key, ref_dog in ref.items():
            day, slot = key.split("_")
            cell = load_cell(photo_dir / f"{key}.jpg")
            if cell is None:
                results[key] = {"pill": None, "score": None}
                continue
            score = cell_score(cell, ref_dog)
            results[key] = {"pill": score > AREA_THRESHOLD,
                            "score": round(score, 4)}
        all_results[photo_dir.name] = results
        n_pills = sum(1 for r in results.values() if r["pill"])
        print(f"{photo_dir.name}: {n_pills}/21 cells with pills")
        if args.annotate:
            out = Path(args.annotate)
            out.mkdir(parents=True, exist_ok=True)
            annotate(photo_dir, results, out / f"{photo_dir.name}.jpg")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
