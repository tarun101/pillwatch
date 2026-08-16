#!/usr/bin/env python3
"""Per-camera calibration profiles for the pillbox grid.

Box-finding and warping were originally tuned as hard-coded constants against
one Raspberry Pi imx708 capture. A second camera (a Jetson's USB/Wyze cam, an
ESP32-CAM) sees the jig at a different resolution, field of view and mounting,
so those constants no longer locate the grid. Rather than fork the code per
camera, the calibration lives in a **profile**: a small JSON file describing the
reference photo, the 21-cell quad, the match anchors and the grid layout.

`crop_cells` loads the active profile at import and exposes its fields under the
same module-level names it always had (``REF_IMAGE``, ``REF_QUAD``, ``ANCHORS``,
``MATCH_SCALE`` …), so every caller keeps working unchanged. The active profile
is chosen by ``PILLBOX_CAMERA_PROFILE`` (default ``imx708``); profiles live in
``detect/profiles/<name>.json``. Generate one for a new camera with
``detect/calibrate_camera.py``.

The ``imx708`` profile reproduces the original constants bit-for-bit, so the
Raspberry Pi deployment and every committed pillbox-data number are unchanged.
"""
import json
import os
from pathlib import Path

PROFILE_DIR = Path(__file__).resolve().parent / "profiles"
DEFAULT_PROFILE = "imx708"
ENV_VAR = "PILLBOX_CAMERA_PROFILE"

# Required keys and the type each must decode to; keeps a hand-written profile
# from silently half-loading.
_REQUIRED = {
    "ref_image": str,
    "ref_quad": list,      # 4 x [x, y]  — TL, TR, BR, BL
    "anchors": list,       # N x [x0, y0, x1, y1]
    "match_scale": (int, float),
    "min_match_confidence": (int, float),
    "cell_w": int,
    "cell_h": int,
    "days": list,          # column order, left->right as the camera sees it
    "slots": list,         # row order, top->bottom
}


class ProfileError(ValueError):
    """A profile file is missing, malformed, or fails validation."""


def profile_path(name):
    return PROFILE_DIR / f"{name}.json"


def available_profiles():
    return sorted(p.stem for p in PROFILE_DIR.glob("*.json"))


def _validate(name, data):
    missing = [k for k in _REQUIRED if k not in data]
    if missing:
        raise ProfileError(f"profile '{name}' missing keys: {', '.join(missing)}")
    for key, typ in _REQUIRED.items():
        if not isinstance(data[key], typ):
            raise ProfileError(
                f"profile '{name}' key '{key}' must be {typ}, got {type(data[key])}")
    if len(data["ref_quad"]) != 4 or any(len(p) != 2 for p in data["ref_quad"]):
        raise ProfileError(f"profile '{name}' ref_quad must be 4 [x, y] points")
    if not data["anchors"] or any(len(a) != 4 for a in data["anchors"]):
        raise ProfileError(
            f"profile '{name}' anchors must be a non-empty list of "
            "[x0, y0, x1, y1] boxes")
    if not (0 < data["match_scale"] <= 1):
        raise ProfileError(f"profile '{name}' match_scale must be in (0, 1]")


def load_profile(name=None):
    """Load a calibration profile by name (env var, else the imx708 default).

    Returns the validated dict. Raises ProfileError with an actionable message
    on a missing or malformed profile rather than importing half-configured.
    """
    if name is None:
        name = os.environ.get(ENV_VAR, DEFAULT_PROFILE).strip() or DEFAULT_PROFILE
    path = profile_path(name)
    if not path.is_file():
        have = ", ".join(available_profiles()) or "(none)"
        raise ProfileError(
            f"camera profile '{name}' not found at {path}. "
            f"Available: {have}. Create one with detect/calibrate_camera.py.")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ProfileError(f"profile '{name}' is not valid JSON: {exc}") from exc
    _validate(name, data)
    data.setdefault("name", name)
    return data


if __name__ == "__main__":
    import sys
    for n in (sys.argv[1:] or available_profiles()):
        try:
            p = load_profile(n)
            print(f"{n:12s} OK  ref={p['ref_image']} "
                  f"cells={len(p['days'])}x{len(p['slots'])} "
                  f"quad={p['ref_quad']}")
        except ProfileError as exc:
            print(f"{n:12s} FAIL  {exc}")
