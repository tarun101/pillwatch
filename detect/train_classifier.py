#!/usr/bin/env python3
"""Train the pill-presence CNN on labelled cell crops and export ONNX.

Labels live in detect/labels.json ("<photo stem>/<DAY>_<SLOT>" -> "pill" |
"empty"); they were bootstrapped from the DoG baseline in classify_cells.py
and hand-reviewed (ambiguous cells were dropped). Crops come from
dataset/cells/ (run detect/crop_cells.py first).

The model is a small from-scratch CNN (~100k params) that sees SIX input
channels: the cell crop plus the same cell from a reference photo of the
empty box. Giving the network the reference to compare against is what makes
low-contrast pills (yellow pill behind the yellow lid) separable — a lone
crop of a tinted lid doesn't carry enough signal, which is also why the DoG
baseline misses them. Augmentation applies the *same* geometric jitter to
both halves (they are aligned crops) but *independent* photometric jitter
(auto-exposure genuinely differs between shots).

Photos taken seconds apart are near-duplicates, so the train/val split is by
capture *scene* (photos within 3s = one scene, and a "<name> copy" file
joins its original) to avoid leakage.

Usage:
    python3 detect/train_classifier.py [--epochs 100] [--val-frac 0.25]
                                       [--out detect/pill_classifier.onnx]

Exports the ONNX model (input: float32 NCHW 1x6x128x96, cell RGB then
reference RGB, in [0,1]; output: logits for [empty, pill]) and copies the 21
reference cell crops to detect/reference_cells/ for inference.
"""
import argparse
import json
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

try:  # works both as `-m detect.train_classifier` and as a bare script
    from . import camera_profiles
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from detect import camera_profiles

IN_H, IN_W = 128, 96  # cell crops are 380x280, similar aspect
# Empty-box reference photo stem follows the active camera profile.
REF_STEM = Path(camera_profiles.load_profile()["ref_image"]).stem
SEED = 0


def scene_of(stem, scenes, gap_s=3):
    """Map photo stem to a scene id; consecutive shots <=gap_s apart share one."""
    base = stem.replace(" copy", "")
    t = datetime.strptime(base, "photo_%Y%m%d_%H%M%S")
    for sid, last in scenes:
        if abs((t - last).total_seconds()) <= gap_s:
            return sid, t
    return None, t


def load_dataset(cells_dir, labels):
    """Returns (images uint8 NHWC-RGB, y, scene_id per sample)."""
    stems = sorted({k.split("/")[0] for k in labels})
    scene_ids, scenes = {}, []  # scenes: list of (sid, last_time)
    for stem in stems:  # stems sorted => chronological
        sid, t = scene_of(stem, scenes)
        if sid is None:
            sid = len({s for s, _ in scenes})
            scenes.append((sid, t))
        else:
            scenes = [(s, t if s == sid else lt) for s, lt in scenes]
        scene_ids[stem] = sid
    refs = {}
    for p in (cells_dir / REF_STEM).glob("*.jpg"):
        img = cv2.resize(cv2.imread(str(p)), (IN_W, IN_H),
                         interpolation=cv2.INTER_AREA)
        refs[p.stem] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    X, y, groups, samp_stems = [], [], [], []
    for key, lab in sorted(labels.items()):
        stem, cell = key.split("/")
        img = cv2.imread(str(cells_dir / stem / f"{cell}.jpg"))
        if img is None:
            continue
        img = cv2.resize(img, (IN_W, IN_H), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        X.append(np.concatenate([img, refs[cell]], axis=2))  # HxWx6
        y.append(1 if lab == "pill" else 0)
        groups.append(scene_ids[stem])
        samp_stems.append(stem)
    return np.stack(X), np.array(y), np.array(groups), np.array(samp_stems)


def photometric(rgb):
    rng = np.random
    out = rgb * rng.uniform(0.75, 1.25)               # brightness
    out = (out - out.mean()) * rng.uniform(0.85, 1.2) + out.mean()  # contrast
    out += rng.uniform(-15, 15, size=3)               # colour cast
    return np.clip(out, 0, 255)


def augment(img6):
    """Same geometric jitter for cell+reference, independent photometric."""
    rng = np.random
    out = img6.astype(np.float32)
    out = np.concatenate([photometric(out[:, :, :3]),
                          photometric(out[:, :, 3:])], axis=2)
    if rng.rand() < 0.5:
        out = out[:, ::-1]                            # horizontal flip
    # small shift/scale/rotation, like re-placing the box slightly off
    m = cv2.getRotationMatrix2D((IN_W / 2, IN_H / 2),
                                rng.uniform(-3, 3), rng.uniform(0.94, 1.06))
    m[:, 2] += rng.uniform(-5, 5, size=2)
    out = cv2.warpAffine(out, m, (IN_W, IN_H), borderMode=cv2.BORDER_REFLECT)
    return out


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        def block(ci, co):
            return [nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co),
                    nn.ReLU(inplace=True), nn.MaxPool2d(2)]
        self.features = nn.Sequential(
            *block(6, 16), *block(16, 32), *block(32, 64), *block(64, 96),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.3), nn.Linear(96, 2),
        )

    def forward(self, x):
        return self.head(self.features(x))


def batches(X, y, idx, bs, train):
    order = np.random.permutation(idx) if train else idx
    for i in range(0, len(order), bs):
        sel = order[i:i + bs]
        imgs = [augment(X[j]) if train else X[j].astype(np.float32) for j in sel]
        xb = torch.from_numpy(np.stack(imgs).transpose(0, 3, 1, 2) / 255.0).float()
        yield xb, torch.from_numpy(y[sel]).long()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cells", default="dataset/cells")
    ap.add_argument("--labels", default="detect/labels.json")
    ap.add_argument("--out", default="detect/pill_classifier.onnx")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--splits-dir", metavar="DIR",
                    help="pillbox-data splits/: train on train.txt, validate on "
                         "valid.txt, and NEVER touch test.txt (held out). Overrides "
                         "the internal scene-based --val-frac split.")
    args = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    labels = json.load(open(args.labels))
    X, y, groups, stems = load_dataset(Path(args.cells), labels)
    if args.splits_dir:
        # Honour the frozen dataset splits so the test set stays untouched.
        split_of = {}
        for name in ("train", "valid", "test"):
            f = Path(args.splits_dir) / f"{name}.txt"
            if f.is_file():
                for s in f.read_text().split():
                    if s.strip():
                        split_of[s.strip()] = name
        tr_idx = np.array([i for i, s in enumerate(stems)
                           if split_of.get(s) == "train"])
        val_idx = np.array([i for i, s in enumerate(stems)
                            if split_of.get(s) == "valid"])
        n_test = sum(1 for s in stems if split_of.get(s) == "test")
        n_unassigned = sum(1 for s in stems if s not in split_of)
        print(f"{len(X)} samples via splits/ -> train {len(tr_idx)} / "
              f"val {len(val_idx)} (held out: {n_test} test, "
              f"{n_unassigned} unassigned cells excluded)")
        if len(tr_idx) == 0 or len(val_idx) == 0:
            sys.exit("error: empty train or valid split — check --splits-dir")
    else:
        uniq = np.unique(groups)
        rng = np.random.RandomState(SEED)
        rng.shuffle(uniq)
        n_val = max(1, int(round(len(uniq) * args.val_frac)))
        val_groups = set(uniq[:n_val])
        val_idx = np.where([g in val_groups for g in groups])[0]
        tr_idx = np.where([g not in val_groups for g in groups])[0]
        print(f"{len(X)} samples, {len(uniq)} scenes -> "
              f"train {len(tr_idx)} / val {len(val_idx)} "
              f"(val pill share {y[val_idx].mean():.2f})")

    model = Net()
    n_par = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_par/1e3:.0f}k")
    w = torch.tensor([1.0, (y[tr_idx] == 0).sum() / max(1, (y[tr_idx] == 1).sum())])
    crit = nn.CrossEntropyLoss(weight=w.float())
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    best_acc, best_state = 0.0, None
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for xb, yb in batches(X, y, tr_idx, args.batch, True):
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(yb)
        sched.step()
        model.eval()
        preds = []
        with torch.no_grad():
            for xb, yb in batches(X, y, val_idx, args.batch, False):
                preds.append(model(xb).argmax(1).numpy())
        pv = np.concatenate(preds)
        acc = (pv == y[val_idx]).mean()
        if acc >= best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"epoch {ep+1:3d}  train loss {tot/len(tr_idx):.4f}  "
              f"val acc {acc:.4f}  (best {best_acc:.4f})")

    model.load_state_dict(best_state)
    model.eval()
    # final confusion on val
    preds = []
    with torch.no_grad():
        for xb, yb in batches(X, y, val_idx, args.batch, False):
            preds.append(model(xb).argmax(1).numpy())
    pv = np.concatenate(preds)
    yv = y[val_idx]
    tp = int(((pv == 1) & (yv == 1)).sum()); tn = int(((pv == 0) & (yv == 0)).sum())
    fp = int(((pv == 1) & (yv == 0)).sum()); fn = int(((pv == 0) & (yv == 1)).sum())
    print(f"val confusion: TP {tp}  TN {tn}  FP {fp}  FN {fn}  "
          f"acc {(tp+tn)/len(yv):.4f}")

    dummy = torch.zeros(1, 6, IN_H, IN_W)
    torch.onnx.export(model, (dummy,), args.out, input_names=["image"],
                      output_names=["logits"],
                      dynamic_axes={"image": {0: "batch"}}, dynamo=False)
    print(f"exported {args.out} ({Path(args.out).stat().st_size/1024:.0f} KB)")
    # ship the reference crops needed to build the 6-channel input at inference
    ref_out = Path(args.out).parent / "reference_cells"
    ref_out.mkdir(exist_ok=True)
    for p in sorted((Path(args.cells) / REF_STEM).glob("*.jpg")):
        shutil.copy(p, ref_out / p.name)
    print(f"copied 21 reference cell crops to {ref_out}/")


if __name__ == "__main__":
    main()
