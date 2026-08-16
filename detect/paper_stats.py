#!/usr/bin/env python3
r"""Generate the PillWatch paper's quantitative results from the pillbox-data set.

One entry point for the paper's numbers and figures:

  table1_dataset.{md,json}   Table I  — photos, labeled cells, Full/Empty,
                             camouflaged cells, split sizes
  model_comparison.{csv,md}  per-model accuracy / precision / recall / F1 /
                             RMSE / params / latency  (the Model-Comparison table)
  metrics.json               everything above, machine-readable
  fig_models.png             grouped bar of accuracy & per-class F1 per model
  fig5_camouflage.png        FIGURE 5 — accuracy vs pill-to-lid colour diff (ΔE)
  fig6_hardware.png          FIGURE 6 — per-model latency (+ board power)

What is portable and what is not
--------------------------------
Accuracy, F1, RMSE, params and the ΔE camouflage curves are *deterministic*:
the numbers this prints on a laptop or in the cloud are the same ones the Pi
produces (bit-level FP differences only ever flip a cell sitting exactly on the
0.5 boundary — at most a cell or two, usually on the DoG baseline). So the
accuracy/camouflage outputs are paper-ready from any machine.

Latency and power are hardware-specific. Run

    python3 -m detect.paper_stats --data ../pillbox-data --hardware

*on the deployment device* (Raspberry Pi 4, Raspberry Pi 5, or NVIDIA Jetson
Orin Nano) to fill Figure 6 and the abstract's "[X] ms". Off-device, latency is
still measured but is only indicative (relative between models). Board power is
measured on Pi 5 and Jetson; Pi 4 runs latency-only because it does not expose
total board-power telemetry.

Usage
-----
    python3 -m detect.paper_stats --data ../pillbox-data --out dataset/paper
        [--split test|all] [--models dog,cnn,yolo] [--hardware] [--latency-reps 5]
"""
import argparse
import csv
import importlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from . import crop_cells
from .camouflage_eval import (BIN_EDGES, BIN_LABELS, bin_index, delta_e,
                              load_reference_cells, pill_lid_colours)
from .hardware_power import classify_device_model, read_power_watts

# name, module (analyze(photo)->{CELL:{pill,...}}), onnx weights (for params), blurb
MODELS = [
    ("DoG", "detect.classify_cells", None,
     "difference-of-Gaussians baseline (no training)"),
    ("Ref-CNN", "detect.pipeline", "detect/pill_classifier.onnx",
     "reference-conditioned 6-channel CNN"),
    ("YOLO", "detect.yolo.detect", "detect/yolo/best.onnx",
     "YOLOv8n classifier (transfer-learned)"),
]
REPO = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_data(data_dir):
    labels = json.loads((data_dir / "labels" / "labels.json").read_text())
    photo_of = {p.stem: p for p in (data_dir / "raw").rglob("photo_*.jpg")}
    split_of = {}
    for name in ("train", "valid", "test"):
        f = data_dir / "splits" / f"{name}.txt"
        if f.is_file():
            for s in f.read_text().split():
                if s.strip():
                    split_of[s.strip()] = name
    return labels, photo_of, split_of


def onnx_param_count(rel):
    """Trainable-parameter count of an ONNX model (sum of initializer sizes)."""
    if rel is None:
        return 0
    try:
        import onnx
        m = onnx.load(str(REPO / rel))
        return int(sum(int(np.prod(init.dims)) for init in m.graph.initializer))
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# run a model over the eval photos: predictions + end-to-end latency
# --------------------------------------------------------------------------- #
def run_model(module_name, photos, latency_reps):
    mod = importlib.import_module(module_name)
    preds = {}          # stem -> {CELL: (pill_bool, conf_float)}
    errors = []
    conf_field = None
    for stem, path in photos.items():
        try:
            res = mod.analyze(path)
        except Exception as exc:  # box not found / bad image
            errors.append((stem, str(exc)))
            continue
        cells = {}
        for cell, r in res.items():
            if conf_field is None:  # discover the score field once (prob/conf/score)
                conf_field = next((k for k in ("prob", "conf", "score")
                                   if k in r), None)
            conf = float(r.get(conf_field, 1.0 if r["pill"] else 0.0))
            cells[cell] = (bool(r["pill"]), conf)
        preds[stem] = cells

    # latency: time analyze() end-to-end, warmup first so model-load isn't counted
    lat_ms = None
    timed = [p for s, p in photos.items() if s in preds]
    if timed:
        mod.analyze(timed[0])  # warmup
        samples = []
        for _ in range(max(1, latency_reps)):
            for path in timed:
                t0 = time.perf_counter()
                mod.analyze(path)
                samples.append((time.perf_counter() - t0) * 1000.0)
        lat_ms = {"per_photo_mean": round(float(np.mean(samples)), 1),
                  "per_photo_median": round(float(np.median(samples)), 1),
                  "per_cell_mean": round(float(np.mean(samples)) / 21.0, 2),
                  "n_timed": len(timed)}
    return preds, errors, lat_ms, conf_field


# --------------------------------------------------------------------------- #
# metrics (Full = positive class)
# --------------------------------------------------------------------------- #
def metrics(y_true, y_pred, y_conf):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    n = tp + tn + fp + fn

    def f1(tp_, fp_, fn_):
        p = tp_ / (tp_ + fp_) if tp_ + fp_ else 0.0
        r = tp_ / (tp_ + fn_) if tp_ + fn_ else 0.0
        return (2 * p * r / (p + r) if p + r else 0.0), p, r

    f1_full, prec_full, rec_full = f1(tp, fp, fn)
    f1_empty, prec_empty, rec_empty = f1(tn, fn, fp)  # negative class as positive
    rmse = float(np.sqrt(np.mean((np.array(y_conf) - y_true) ** 2)))
    return {
        "n": n, "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "accuracy": round((tp + tn) / n, 4) if n else None,
        "precision_full": round(prec_full, 4), "recall_full": round(rec_full, 4),
        "f1_full": round(f1_full, 4),
        "precision_empty": round(prec_empty, 4), "recall_empty": round(rec_empty, 4),
        "f1_empty": round(f1_empty, 4),
        "macro_f1": round((f1_full + f1_empty) / 2, 4),
        "rmse": round(rmse, 4),
    }


# --------------------------------------------------------------------------- #
# camouflage ΔE per Full cell (reuses camouflage_eval extraction)
# --------------------------------------------------------------------------- #
def compute_delta_e(full_keys, photo_of, ref_cells):
    templates = crop_cells.build_matcher(REPO / "images" / crop_cells.REF_IMAGE)
    de = {}
    per_photo = {}
    for k in full_keys:
        per_photo.setdefault(k.split("/")[0], []).append(k.split("/", 1)[1])
    for stem, cells in per_photo.items():
        path = photo_of.get(stem)
        if path is None:
            continue
        img = cv2.imread(str(path))
        if img is None:
            continue
        quad, _ = crop_cells.align_quad(img, templates)
        if quad is None:
            continue
        grid = crop_cells.warp_grid(img, quad)
        crops = {f"{d}_{s}": c for d, s, c in crop_cells.cell_crops(grid)}
        for cell in cells:
            lid, pill, _ = pill_lid_colours(crops[cell], ref_cells[cell])
            d2000, _ = delta_e(pill, lid)
            de[f"{stem}/{cell}"] = round(d2000, 3)
    return de


# --------------------------------------------------------------------------- #
# hardware sampling (Pi 4 latency; Pi 5/Jetson latency + power)
# --------------------------------------------------------------------------- #
def measure_power(module_name, photos, seconds=8.0):
    """Average board power while a model runs analyze() in a loop, minus idle."""
    from threading import Event, Thread
    base, source = read_power_watts(with_source=True)
    if base is None:
        return None
    idle_samples = [read_power_watts() for _ in range(5)]
    idle_samples = [w for w in idle_samples if w is not None]
    if not idle_samples:
        return None
    idle = float(np.mean(idle_samples))
    mod = importlib.import_module(module_name)
    paths = list(photos.values())
    if not paths:
        return None
    mod.analyze(paths[0])  # warmup
    samples, stop = [], Event()

    def poll():
        while not stop.wait(0.1):
            w = read_power_watts()
            if w is not None:
                samples.append(w)
    t = Thread(target=poll, daemon=True)
    t.start()
    t0 = time.perf_counter()
    i = 0
    while time.perf_counter() - t0 < seconds:
        mod.analyze(paths[i % len(paths)])
        i += 1
    stop.set()
    t.join(timeout=1)
    if not samples:
        return None
    avg = float(np.mean(samples))
    return {"avg_watts": round(avg, 2), "idle_watts": round(float(idle), 2),
            "net_watts": round(max(0.0, avg - idle), 2), "source": source}


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True, help="pillbox-data checkout")
    ap.add_argument("--out", default="dataset/paper")
    ap.add_argument("--ref-cells", default="detect/reference_cells")
    ap.add_argument("--split", default="test", choices=["test", "valid", "all"],
                    help="evaluate on this split (default: held-out test)")
    ap.add_argument("--models", default="dog,cnn,yolo")
    ap.add_argument("--latency-reps", type=int, default=5)
    ap.add_argument("--hardware", action="store_true",
                    help=("collect deployment-device metrics (Pi 5/Jetson power; "
                          "Pi 4 latency-only) for Figure 6"))
    ap.add_argument("--camo-max-de", type=float, default=None,
                    help="ΔE pinned to 100%% in the pct column (default: dataset max)")
    args = ap.parse_args()

    data = Path(args.data).expanduser()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    labels, photo_of, split_of = load_data(data)

    want = {"dog": "DoG", "cnn": "Ref-CNN", "yolo": "YOLO"}
    chosen = {want[m.strip()] for m in args.models.split(",") if m.strip() in want}
    models = [m for m in MODELS if m[0] in chosen]

    # cells to evaluate on
    def in_eval(stem):
        return args.split == "all" or split_of.get(stem) == args.split
    eval_keys = sorted(k for k in labels if in_eval(k.split("/")[0])
                       and k.split("/")[0] in photo_of)
    eval_stems = sorted({k.split("/")[0] for k in eval_keys})
    photos = {s: photo_of[s] for s in eval_stems}
    print(f"eval split '{args.split}': {len(eval_stems)} photos, "
          f"{len(eval_keys)} labeled cells")

    # ---- Table I -----------------------------------------------------------
    ref_cells = load_reference_cells(Path(args.ref_cells))
    full_keys = [k for k in labels if labels[k] == "pill"]
    de_all = compute_delta_e(full_keys, photo_of, ref_cells)
    camo_de = 2.3  # first ΔE bin edge == "just noticeable difference"
    n_camo = sum(1 for v in de_all.values() if v < camo_de)
    from collections import Counter
    cls = Counter(labels.values())
    table1 = {
        "photographs_captured": len(photo_of),
        "photographs_labeled": len({k.split("/")[0] for k in labels}),
        "labeled_cells": len(labels),
        "full_cells": cls.get("pill", 0), "empty_cells": cls.get("empty", 0),
        "camouflaged_full_cells": n_camo, "camouflage_de_threshold": camo_de,
        "splits_photos": {s: len({k for k, v in split_of.items() if v == s})
                          for s in ("train", "valid", "test")},
    }
    json.dump(table1, open(out / "table1_dataset.json", "w"), indent=2)
    with open(out / "table1_dataset.md", "w") as f:
        f.write("# Table I — Dataset summary\n\n| quantity | value |\n|---|---|\n")
        f.write(f"| Photographs captured | {table1['photographs_captured']} |\n")
        f.write(f"| Photographs labeled | {table1['photographs_labeled']} |\n")
        f.write(f"| Labeled cells | {table1['labeled_cells']} |\n")
        f.write(f"| — Full | {table1['full_cells']} |\n")
        f.write(f"| — Empty | {table1['empty_cells']} |\n")
        f.write(f"| Camouflaged Full cells (ΔE < {camo_de}) | {n_camo} |\n")
        for s in ("train", "valid", "test"):
            f.write(f"| Split: {s} (photos) | {table1['splits_photos'][s]} |\n")
    print("wrote Table I")

    # ---- per-model predictions, metrics, latency, power --------------------
    results = {}
    for name, module_name, weights, blurb in models:
        print(f"running {name} ...")
        preds, errors, lat, conf_field = run_model(module_name, photos,
                                                    args.latency_reps)
        y_true, y_pred, y_conf = [], [], []
        for k in eval_keys:
            stem, cell = k.split("/")
            if stem in preds and cell in preds[stem]:
                pill, conf = preds[stem][cell]
                y_true.append(1 if labels[k] == "pill" else 0)
                y_pred.append(1 if pill else 0)
                y_conf.append(conf)
        m = metrics(y_true, y_pred, y_conf)
        power = (measure_power(module_name, photos) if args.hardware else None)
        results[name] = {"blurb": blurb, "params": onnx_param_count(weights),
                         "metrics": m, "latency_ms": lat, "power": power,
                         "n_align_fail": len(errors), "conf_field": conf_field}
        print(f"  acc={m['accuracy']} F1(full)={m['f1_full']} "
              f"F1(empty)={m['f1_empty']} lat={lat and lat['per_photo_mean']}ms")

    json.dump({"split": args.split, "table1": table1, "models": results,
               "host": _host_desc()}, open(out / "metrics.json", "w"), indent=2)
    write_notes(out, results, args)
    write_model_table(out, results)
    write_figures(out, results, full_keys, de_all, labels, photos, models,
                  args)
    print(f"\nwrote outputs under {out}/")


def _host_desc():
    import platform
    model = None
    try:
        model = Path("/proc/device-tree/model").read_bytes().rstrip(b"\0").decode()
    except (OSError, UnicodeError):
        pass
    device = classify_device_model(model)
    return {"machine": platform.machine(), "processor": platform.processor(),
            "python": platform.python_version(), "device": device,
            "device_model": model}


def write_notes(out, results, args):
    """Self-documenting caveats so paper numbers aren't misread."""
    host = _host_desc()
    on_deployment_device = host["device"] in (
        "raspberry_pi", "raspberry_pi_4", "raspberry_pi_5", "jetson",
    )
    device_label = host["device_model"] or host["machine"]
    power_sources = {r["power"]["source"] for r in results.values()
                     if r["power"] and r["power"].get("source")}
    if "external_meter" in power_sources:
        power_note = ("measured from an external meter via "
                      "PILLBOX_POWER_METER_CMD (net of idle).")
    elif "jetson_tegrastats" in power_sources:
        power_note = "measured from Jetson tegrastats (net of idle)."
    elif "raspberry_pi_pmic" in power_sources:
        power_note = "measured from the Pi 5 PMIC (net of idle)."
    elif host["device"] == "raspberry_pi_4":
        power_note = (
            "not measured: Raspberry Pi 4 has no supported onboard total-board "
            "power telemetry. Latency results are still valid."
        )
    elif host["device"] == "jetson":
        power_note = (
            "not measured: this Jetson exposes no tegrastats power rails (the "
            "Orin Nano/NX drop the onboard INA3221 sensors). Set "
            "PILLBOX_POWER_METER_CMD to an external meter for board power; "
            "latency results are still valid."
        )
    else:
        power_note = ("not measured (needs the Pi 5 PMIC, Jetson tegrastats, or "
                      "an external meter via PILLBOX_POWER_METER_CMD). "
                      "Run `--hardware` on the deployment device.")
    lines = [
        "# How to read these numbers\n",
        f"- **Eval set:** `{args.split}` split.  Accuracy / precision / recall /",
        "  F1 / RMSE / ΔE are deterministic — identical on Pi, Mac or cloud",
        "  (bit-level FP differences only flip a cell exactly on the 0.5 boundary).",
        "  **These are paper-ready from any machine.**\n",
        f"- **Latency:** measured on `{device_label}`.",
        ("  This IS the deployment device — use directly for Figure 6 / the abstract."
         if on_deployment_device else
         "  This is NOT a recognized Pi 4, Pi 5, or Jetson — treat as indicative "
         "(relative ordering only). Re-run with `--hardware` **on the deployment "
         "device** for Figure 6 and the abstract's per-photo time."),
        "",
        "- **Power:** " + power_note,
        "",
        "- **Model provenance caveat:** DoG is untrained, so its numbers are a",
        "  clean held-out result. The shipped Ref-CNN and YOLO were trained on",
        "  subsets of this data *before* the current splits existed, so their",
        "  numbers may include in-sample cells — treat as indicative until models",
        "  with a recorded train/test split (see pillbox-data `models/*/card.json`)",
        "  are dropped in and evaluated on the frozen `test` split.",
        "",
        f"- **Alignment overhead:** the per-photo latency is the full pipeline",
        "  (locate + warp + classify all 21 cells). Box alignment is a shared",
        "  front-end, so it inflates all models by roughly the same amount.",
        "",
    ]
    (out / "README.md").write_text("\n".join(lines))


def write_model_table(out, results):
    cols = ["accuracy", "precision_full", "recall_full", "f1_full",
            "f1_empty", "macro_f1", "rmse"]
    with open(out / "model_comparison.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model"] + cols + ["params", "latency_ms_per_photo",
                                       "net_watts"])
        for name, r in results.items():
            m = r["metrics"]
            w.writerow([name] + [m[c] for c in cols]
                       + [r["params"],
                          r["latency_ms"] and r["latency_ms"]["per_photo_mean"],
                          r["power"] and r["power"]["net_watts"]])
    with open(out / "model_comparison.md", "w") as f:
        f.write("# Model comparison\n\n")
        f.write("| Model | Acc | Prec(Full) | Rec(Full) | F1(Full) | F1(Empty) | "
                "macro-F1 | RMSE | Params | ms/photo | ΔW |\n")
        f.write("|" + "---|" * 11 + "\n")
        for name, r in results.items():
            m = r["metrics"]
            p = f"{r['params']/1e3:.0f}k" if r["params"] else "—"
            lat = r["latency_ms"]["per_photo_mean"] if r["latency_ms"] else "—"
            w = r["power"]["net_watts"] if r["power"] else "—"
            f.write(f"| {name} | {m['accuracy']} | {m['precision_full']} | "
                    f"{m['recall_full']} | {m['f1_full']} | {m['f1_empty']} | "
                    f"{m['macro_f1']} | {m['rmse']} | {p} | {lat} | {w} |\n")


def write_figures(out, results, full_keys, de_all, labels, photos, models, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(results)

    # fig_models: accuracy & per-class F1
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(names))
    w = 0.25
    for i, (key, lab) in enumerate([("accuracy", "Accuracy"),
                                    ("f1_full", "F1 (Full)"),
                                    ("f1_empty", "F1 (Empty)")]):
        ax.bar(x + (i - 1) * w, [results[n]["metrics"][key] for n in names],
               w, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylim(0, 1.02); ax.set_ylabel("score")
    ax.set_title(f"Model comparison ({args.split} split)")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    for i, n in enumerate(names):
        ax.annotate(f"n={results[n]['metrics']['n']}", (x[i], 0),
                    textcoords="offset points", xytext=(0, -28), ha="center",
                    fontsize=8, color="gray")
    fig.tight_layout(); fig.savefig(out / "fig_models.png", dpi=150); plt.close(fig)

    # fig5: camouflage — accuracy (recall on Full cells) vs ΔE bin, per model.
    # Full cells only, so "accuracy" == correctly-detected fraction == recall.
    preds_full = {}  # name -> {stem/cell: pill_bool} (recompute cheaply from run)
    # We reuse each model's analyze already ran during metrics? Not stored per-cell
    # globally, so recompute predictions for the Full cells here.
    templates = None
    for name, module_name, *_ in models:
        mod = importlib.import_module(module_name)
        pf = {}
        cache = {}
        for k in full_keys:
            stem, cell = k.split("/")
            if stem not in photos:
                continue
            if stem not in cache:
                try:
                    cache[stem] = mod.analyze(photos[stem])
                except Exception:
                    cache[stem] = None
            res = cache[stem]
            if res and cell in res:
                pf[k] = bool(res[cell]["pill"])
        preds_full[name] = pf

    n_bins = len(BIN_LABELS)
    counts = [0] * n_bins
    hit = {n: [0] * n_bins for n in preds_full}
    for k, de in de_all.items():
        if k not in labels or labels[k] != "pill":
            continue
        if k.split("/")[0] not in photos:
            continue  # only cells actually run this split — keep num/denom aligned
        bi = bin_index(de)
        counts[bi] += 1
        for n in preds_full:
            if preds_full[n].get(k):
                hit[n][bi] += 1
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    xs = np.arange(n_bins)
    for n in preds_full:
        ys = [hit[n][b] / counts[b] if counts[b] else np.nan for b in range(n_bins)]
        ax.plot(xs, ys, marker="o", label=n)
    ax.set_xticks(xs); ax.set_xticklabels(BIN_LABELS)
    ax.set_xlabel("pill-to-lid colour difference  ΔE (CIEDE2000)  —  low = camouflaged")
    ax.set_ylabel("accuracy on Full cells (recall)")
    ax.set_ylim(0, 1.02); ax.grid(True, alpha=0.3)
    ax.set_title("Figure 5 — Effect of camouflage: accuracy vs pill-to-lid ΔE")
    ax.legend()
    for b in range(n_bins):
        ax.annotate(f"n={counts[b]}", (xs[b], 0), textcoords="offset points",
                    xytext=(0, -30), ha="center", fontsize=8, color="gray")
    fig.tight_layout(); fig.savefig(out / "fig5_camouflage.png", dpi=150)
    plt.close(fig)
    json.dump({"bin_labels": BIN_LABELS, "n_full": counts,
               "recall": {n: [round(hit[n][b] / counts[b], 4) if counts[b] else None
                              for b in range(n_bins)] for n in preds_full}},
              open(out / "fig5_camouflage.json", "w"), indent=2)

    # fig6: hardware — per-model latency (+ power if measured)
    have_power = any(results[n]["power"] for n in names)
    fig, axes = plt.subplots(1, 2 if have_power else 1,
                             figsize=(9 if have_power else 5, 4.2), squeeze=False)
    lat = [results[n]["latency_ms"]["per_photo_mean"] if results[n]["latency_ms"]
           else 0 for n in names]
    axes[0][0].bar(names, lat, color="#4477aa")
    axes[0][0].set_ylabel("inference time (ms / photo)")
    host = _host_desc()
    if have_power:
        latency_suffix = ""
    elif args.hardware and host["device"] == "raspberry_pi_4":
        latency_suffix = "  (Pi 4; board power unavailable)"
    else:
        latency_suffix = "  (host — run on device for Fig 6)"
    axes[0][0].set_title("Latency" + latency_suffix)
    for i, v in enumerate(lat):
        axes[0][0].annotate(f"{v:.0f}", (i, v), ha="center", va="bottom", fontsize=9)
    if have_power:
        pw = [results[n]["power"]["net_watts"] if results[n]["power"] else 0
              for n in names]
        axes[0][1].bar(names, pw, color="#66aa55")
        axes[0][1].set_ylabel("net power draw (W, above idle)")
        axes[0][1].set_title("Power")
    fig.suptitle("Figure 6 — Hardware cost of each model")
    fig.tight_layout(); fig.savefig(out / "fig6_hardware.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
