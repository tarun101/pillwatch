#!/usr/bin/env python3
"""Crop the 21 pillbox cells out of full-resolution captures.

The camera and pillbox sit in a fixed jig, so the cell grid was calibrated
once against a reference photo (REF_IMAGE + REF_QUAD below). Each new photo
is aligned to the reference by template-matching two anchor patches (left and
right halves of the box), which recovers the small translation/rotation
differences from re-placing the box. The calibrated grid is then warped to a
canonical top-down view and sliced into 7x3 = 21 cell images.

Usage:
    python3 detect/crop_cells.py [--images DIR] [--out DIR] [--debug]

Outputs one folder per photo under --out (default: dataset/cells/):
    dataset/cells/<photo stem>/<DAY>_<SLOT>.jpg   e.g. MON_MORN.jpg

With --debug, also writes <photo stem>_grid.jpg overlays under --out/.debug/
showing where the grid landed, for visual QA.
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Worker-thread knob for the detectors. Default 0 = use all cores (fastest).
# On a marginal power supply the resulting CPU burst can brown out the Pi —
# set PILLBOX_THREADS=1 or 2 in the service environment to trade speed for a
# lower peak draw. The ONNX detectors read CPU_THREADS too.
_req = os.environ.get("PILLBOX_THREADS", "0")
CPU_THREADS = int(_req) if _req.strip() else 0
cv2.setNumThreads(CPU_THREADS)  # 0 -> OpenCV uses all cores

# ---- calibration (relative to REF_IMAGE, full-resolution pixels) ----------
# These constants come from the active camera profile (detect/profiles/*.json,
# selected by PILLBOX_CAMERA_PROFILE, default "imx708"). The imx708 profile
# reproduces the original hand-tuned values exactly, so the Pi deployment is
# unchanged; a second camera (Jetson/Wyze, ESP32-CAM) sets its own profile.
# Names below are kept identical so every caller (crop/align/warp) is untouched.
try:  # works both as `-m detect.crop_cells` and `python3 detect/crop_cells.py`
    from . import camera_profiles
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from detect import camera_profiles

_PROFILE = camera_profiles.load_profile()
REF_IMAGE = _PROFILE["ref_image"]
# Corners of the 21-cell area: TL, TR, BR, BL.
REF_QUAD = np.float32(_PROFILE["ref_quad"])
# Anchor patches used to re-locate the box in new photos (x0, y0, x1, y1).
# Two+ patches (e.g. left / right half of the box) let us recover rotation too.
ANCHORS = [tuple(a) for a in _PROFILE["anchors"]]
MATCH_SCALE = float(_PROFILE["match_scale"])   # template matching downscale
MIN_MATCH_CONFIDENCE = float(_PROFILE["min_match_confidence"])

# Column order as seen by the camera (labels face away from it), left to
# right; row order top to bottom.
DAYS = list(_PROFILE["days"])
SLOTS = list(_PROFILE["slots"])

# Canonical warped size of one cell.
CELL_W, CELL_H = int(_PROFILE["cell_w"]), int(_PROFILE["cell_h"])
GRID_W, GRID_H = CELL_W * len(DAYS), CELL_H * len(SLOTS)


def build_matcher(ref_path):
    """Precompute downscaled anchor templates from the reference image."""
    ref = cv2.imread(str(ref_path))
    if ref is None:
        sys.exit(f"error: cannot read reference image {ref_path}")
    small = cv2.resize(ref, None, fx=MATCH_SCALE, fy=MATCH_SCALE,
                       interpolation=cv2.INTER_AREA)
    templates = []
    for x0, y0, x1, y1 in ANCHORS:
        s = MATCH_SCALE
        tpl = small[int(y0 * s):int(y1 * s), int(x0 * s):int(x1 * s)]
        center = np.float32([(x0 + x1) / 2, (y0 + y1) / 2])
        templates.append((tpl, center))
    return templates


def locate_anchor(small, tpl):
    """Return (peak position in full-res coords, confidence)."""
    res = cv2.matchTemplate(small, tpl, cv2.TM_CCOEFF_NORMED)
    _, conf, _, loc = cv2.minMaxLoc(res)
    th, tw = tpl.shape[:2]
    center = np.float32([(loc[0] + tw / 2) / MATCH_SCALE,
                         (loc[1] + th / 2) / MATCH_SCALE])
    return center, conf


def align_quad(img, templates):
    """Locate the cell grid in img. Returns (quad, per-anchor confidences)."""
    small = cv2.resize(img, None, fx=MATCH_SCALE, fy=MATCH_SCALE,
                       interpolation=cv2.INTER_AREA)
    src_pts, dst_pts, confs = [], [], []
    for tpl, ref_center in templates:
        center, conf = locate_anchor(small, tpl)
        confs.append(conf)
        if conf >= MIN_MATCH_CONFIDENCE:
            src_pts.append(ref_center)
            dst_pts.append(center)
    if len(src_pts) == 2:
        # similarity transform (translation + rotation + scale) from 2 points
        M, _ = cv2.estimateAffinePartial2D(np.float32([src_pts]),
                                           np.float32([dst_pts]))
    elif len(src_pts) == 1:
        off = dst_pts[0] - src_pts[0]
        M = np.float32([[1, 0, off[0]], [0, 1, off[1]]])
    else:
        return None, confs
    quad = cv2.transform(REF_QUAD.reshape(1, -1, 2), M).reshape(-1, 2)
    return np.float32(quad), confs


def warp_grid(img, quad):
    """Warp the quad to a canonical top-down GRID_W x GRID_H image."""
    dst = np.float32([[0, 0], [GRID_W, 0], [GRID_W, GRID_H], [0, GRID_H]])
    H = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(img, H, (GRID_W, GRID_H))


def cell_crops(grid_img):
    """Yield (day, slot, crop) for all 21 cells of a warped grid image."""
    for r, slot in enumerate(SLOTS):
        for c, day in enumerate(DAYS):
            crop = grid_img[r * CELL_H:(r + 1) * CELL_H,
                            c * CELL_W:(c + 1) * CELL_W]
            yield day, slot, crop


def draw_debug(img, quad):
    dbg = img.copy()
    ncol, nrow = len(DAYS), len(SLOTS)
    cv2.polylines(dbg, [quad.astype(int)], True, (0, 0, 255), 6)
    dst = np.float32([[0, 0], [ncol, 0], [ncol, nrow], [0, nrow]])
    H = cv2.getPerspectiveTransform(dst, quad)
    for c in range(1, ncol):
        p = cv2.perspectiveTransform(np.float32([[[c, 0]], [[c, nrow]]]), H).astype(int)
        cv2.line(dbg, tuple(p[0, 0]), tuple(p[1, 0]), (0, 0, 255), 3)
    for r in range(1, nrow):
        p = cv2.perspectiveTransform(np.float32([[[0, r]], [[ncol, r]]]), H).astype(int)
        cv2.line(dbg, tuple(p[0, 0]), tuple(p[1, 0]), (0, 0, 255), 3)
    return cv2.resize(dbg, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--images", default="images", help="input photo directory")
    ap.add_argument("--out", default="dataset/cells", help="output directory")
    ap.add_argument("--debug", action="store_true",
                    help="write grid overlays to <out>/.debug/")
    args = ap.parse_args()

    image_dir = Path(args.images)
    out_dir = Path(args.out)
    ref_path = image_dir / REF_IMAGE
    templates = build_matcher(ref_path)

    photos = sorted(image_dir.glob("*.jpg"))
    if not photos:
        sys.exit(f"error: no .jpg files in {image_dir}")
    failed = []
    for p in photos:
        img = cv2.imread(str(p))
        if img is None:
            failed.append((p.name, "unreadable"))
            continue
        quad, confs = align_quad(img, templates)
        conf_str = "/".join(f"{c:.2f}" for c in confs)
        if quad is None:
            failed.append((p.name, f"box not found (conf {conf_str})"))
            continue
        grid = warp_grid(img, quad)
        dest = out_dir / p.stem
        dest.mkdir(parents=True, exist_ok=True)
        for day, slot, crop in cell_crops(grid):
            cv2.imwrite(str(dest / f"{day}_{slot}.jpg"), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
        if args.debug:
            dbg_dir = out_dir / ".debug"
            dbg_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dbg_dir / f"{p.stem}_grid.jpg"),
                        draw_debug(img, quad))
        print(f"{p.name}: ok (conf {conf_str})")
    for name, why in failed:
        print(f"{name}: FAILED — {why}", file=sys.stderr)
    print(f"\n{len(photos) - len(failed)}/{len(photos)} photos cropped -> {out_dir}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
