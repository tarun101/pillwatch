#!/usr/bin/env python3
"""Generate a camera calibration profile from one empty-box photo.

Detection is calibrated to a camera's geometry (see detect/camera_profiles.py).
When you mount a new camera — a Jetson's USB/Wyze cam, an ESP32-CAM — take one
photo of the *empty* pillbox in its final position and run this tool to produce
``detect/profiles/<name>.json``. From then on, select it with
``PILLBOX_CAMERA_PROFILE=<name>`` and the whole detection stack (DoG, CNN, YOLO)
works against the new camera.

You supply the four corners of the 21-cell area (TL, TR, BR, BL, in pixels).
The two match anchors — the patches the aligner template-matches to re-locate
the box in later photos — are derived automatically as the left and right
portions of that quad, so you only ever mark four points.

Two ways to give the corners:

  # interactive: click TL, TR, BR, BL in a window (needs a display)
  python3 -m detect.calibrate_camera --photo empty.jpg --name jetson_wyze --click

  # headless: pass the four corners explicitly
  python3 -m detect.calibrate_camera --photo empty.jpg --name jetson_wyze \
      --quad 640,180 1890,175 1900,1010 630,1015

After writing the profile, verify it before trusting any numbers:

  PILLBOX_CAMERA_PROFILE=jetson_wyze python3 detect/crop_cells.py \
      --images <dir-with-empty.jpg> --out /tmp/cal --debug
  # inspect /tmp/cal/.debug/*_grid.jpg — the red grid must sit on the cells

The empty photo also becomes the CNN/DoG reference: copy it into images/ under
the name recorded in the profile (the tool does this for you with --copy-into
images) and regenerate reference crops with crop_cells.py.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_DAYS = ["SAT", "FRI", "THU", "WED", "TUE", "MON", "SUN"]
DEFAULT_SLOTS = ["NIGHT", "NOON", "MORN"]


def _parse_point(text):
    try:
        x, y = text.split(",")
        return [int(round(float(x))), int(round(float(y)))]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"corner '{text}' must be x,y (e.g. 640,180)")


def anchors_from_quad(quad, frac=0.44, pad_frac=0.02):
    """Left/right anchor boxes covering the quad, mirroring the imx708 layout.

    Each anchor is a full-height strip spanning ~frac of the box width from one
    side; a small vertical pad makes matching robust to placement wobble. Two
    strips on opposite sides let the aligner recover rotation, not just shift.
    """
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    w, h = xmax - xmin, ymax - ymin
    pad = int(h * pad_frac)
    y0, y1 = ymin, ymax + pad
    left = [xmin, y0, int(xmin + frac * w), y1]
    right = [int(xmax - frac * w), y0, xmax, y1]
    return [left, right]


def click_quad(photo_path):
    """Let the user click the four corners TL, TR, BR, BL in a window."""
    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("--click needs matplotlib (pip install matplotlib), or pass "
                 "--quad instead")
    import cv2
    img = cv2.imread(str(photo_path))
    if img is None:
        sys.exit(f"cannot read photo {photo_path}")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    labels = ["TL (top-left)", "TR (top-right)",
              "BR (bottom-right)", "BL (bottom-left)"]
    pts = []
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.imshow(rgb)
    ax.set_title(f"click 4 corners in order: {labels[0]}")

    def onclick(event):
        if event.xdata is None:
            return
        pts.append([int(round(event.xdata)), int(round(event.ydata))])
        ax.plot(event.xdata, event.ydata, "r+", markersize=14, mew=2)
        if len(pts) < 4:
            ax.set_title(f"click 4 corners in order: {labels[len(pts)]}")
        else:
            ax.set_title("done — close the window")
        fig.canvas.draw()
        if len(pts) == 4:
            fig.canvas.mpl_disconnect(cid)

    cid = fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()
    if len(pts) != 4:
        sys.exit(f"need 4 corners, got {len(pts)}")
    return pts


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--photo", required=True, help="one photo of the EMPTY box")
    ap.add_argument("--name", required=True, help="profile name, e.g. jetson_wyze")
    ap.add_argument("--quad", nargs=4, type=_parse_point, metavar="x,y",
                    help="the 4 corners TL TR BR BL (pixels); or use --click")
    ap.add_argument("--click", action="store_true",
                    help="click the 4 corners interactively (needs a display)")
    ap.add_argument("--days", nargs="+", default=DEFAULT_DAYS,
                    help="column labels left->right as the camera sees them")
    ap.add_argument("--slots", nargs="+", default=DEFAULT_SLOTS,
                    help="row labels top->bottom")
    ap.add_argument("--cell", nargs=2, type=int, default=[280, 380],
                    metavar=("W", "H"), help="warped cell size in px")
    ap.add_argument("--match-scale", type=float, default=0.25)
    ap.add_argument("--min-confidence", type=float, default=0.55)
    ap.add_argument("--anchor-frac", type=float, default=0.44,
                    help="fraction of box width each side anchor spans")
    ap.add_argument("--copy-into", metavar="DIR",
                    help="also copy the empty photo here (e.g. images/) so it "
                         "serves as the DoG/CNN reference")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing profile of the same name")
    args = ap.parse_args()

    photo = Path(args.photo)
    if not photo.is_file():
        sys.exit(f"photo not found: {photo}")
    if args.click and args.quad:
        sys.exit("pass either --quad or --click, not both")
    quad = args.quad if args.quad else (click_quad(photo) if args.click else None)
    if quad is None:
        sys.exit("supply the corners with --quad x,y x,y x,y x,y or --click")

    profile = {
        "name": args.name,
        "description": f"Calibrated from {photo.name} by calibrate_camera.py",
        "ref_image": photo.name,
        "ref_quad": [list(p) for p in quad],
        "anchors": anchors_from_quad(quad, frac=args.anchor_frac),
        "match_scale": args.match_scale,
        "min_match_confidence": args.min_confidence,
        "cell_w": args.cell[0],
        "cell_h": args.cell[1],
        "days": args.days,
        "slots": args.slots,
    }

    # Validate through the same loader every caller uses before writing.
    from detect import camera_profiles
    camera_profiles._validate(args.name, profile)

    out = camera_profiles.profile_path(args.name)
    if out.exists() and not args.force:
        sys.exit(f"{out} exists; pass --force to overwrite")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(profile, indent=2) + "\n")
    print(f"wrote {out}")
    print(f"  ref_image: {profile['ref_image']}")
    print(f"  quad:      {profile['ref_quad']}")
    print(f"  anchors:   {profile['anchors']}")

    if args.copy_into:
        dest_dir = Path(args.copy_into)
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(photo, dest_dir / photo.name)
        print(f"copied reference photo -> {dest_dir / photo.name}")

    print("\nnext:")
    print(f"  PILLBOX_CAMERA_PROFILE={args.name} python3 detect/crop_cells.py \\")
    print("      --images <dir-with-your-empty-photo> --out /tmp/cal --debug")
    print("  # check /tmp/cal/.debug/*_grid.jpg: the red grid must land on the cells")


if __name__ == "__main__":
    main()
