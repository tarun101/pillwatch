# On-device pill inference on the ESP32-S3

Phase-2 goal: run PillWatch's detectors **on the ESP32-S3 itself**, not just use
it as a camera. This directory converts the reference-conditioned CNN to int8
TFLite and runs it on-device with [esp-tflite-micro], reporting latency, memory
and the prediction over serial.

> **Status:** reference implementation. The Python conversion and the model-fit
> analysis below are grounded in the actual shipped models; the firmware is
> written against the esp-tflite-micro API but has **not been compiled or run on
> hardware here** — it needs a board to validate (arena size, op set, and the
> onnx2tf op mapping are the likely tuning points).

## Which models fit? (measured from the shipped ONNX)

| Model | Params | int8 size | Input | Verdict on ESP32-S3 (240 MHz, 8 MB PSRAM, no NPU) |
|---|---|---|---|---|
| **Ref-CNN** | 79.6 k | **~78 KB** | 6×128×96 | **Primary target.** All standard ops (Conv/MaxPool/ReLU/GAP/FC). Fits arena in PSRAM with room to spare. |
| DoG baseline | 0 (classical) | — | per cell | **Portable to C** — difference-of-Gaussians is a few separable blurs + a threshold; no framework needed. Cheapest on-device option. |
| YOLO classifier | 1.44 M | ~1.4 MB | 3×224×224 | **Stretch.** Fits PSRAM, but 224² convs + SiLU (Sigmoid×26/Mul×26) on a 240 MHz core will be seconds/inference. Only if the paper needs it. |

So the realistic on-device ladder is **DoG (hand-C) → Ref-CNN (this firmware) →
YOLO (stretch)**. This directory implements the Ref-CNN rung.

## Scope of the firmware

It classifies **one** 6-channel cell (cell RGB + empty-reference RGB, 128×96) —
the CNN's native input. Locating and cropping the 21 cells from a full frame
still uses the OpenCV front-end (`detect/crop_cells.py`) that runs off-board;
porting box-detection to the ESP32 is a separate task. The harness runs on a
fixed test pattern so the latency/memory numbers are reproducible; wiring the
OV5640 capture into a per-cell buffer is the marked next step in `main.cpp`.

## Build & run

**1. Generate the model header** (off-device, in a desktop/CI env with TF):

```bash
pip install tensorflow onnx2tf onnx onnx-graphsurgeon numpy opencv-python-headless
# a representative set for int8 calibration — real cells + their empty references
python3 pillbox/detect/export_dataset.py --data pillbox-data --out /tmp/exp
python3 firmware/esp32s3_cnn_inference/convert_cnn_to_tflite.py \
    --onnx detect/pill_classifier.onnx \
    --cells /tmp/exp --reference detect/reference_cells \
    --out firmware/esp32s3_cnn_inference/main/model_data.h
```

`model_data.h` is git-ignored — regenerate it whenever the CNN is retrained.

**2. Build & flash** (ESP-IDF ≥ 5.1):

```bash
cd firmware/esp32s3_cnn_inference
idf.py set-target esp32s3
idf.py build flash monitor      # esp-tflite-micro is pulled as a managed component
```

Expected serial output:

```
=== PillWatch CNN on ESP32-S3 ===
model:        ~80000 bytes int8
latency:      NN.N ms / inference (avg of 50)
arena used:   NNN KB of 512 KB
PSRAM free:   ...
prediction:   EMPTY (logits empty=... pill=...)
```

## Validate int8 accuracy before trusting it

Quantization can shift predictions. After converting, compare the int8 TFLite
model against the float ONNX on the held-out test split and confirm the drop is
small (a point or two of macro-F1 is normal):

```bash
# float reference:
python3 -m detect.paper_stats --data pillbox-data --split test --models cnn
# then evaluate the .tflite the same way (host-side TFLite interpreter) and diff.
```

If the int8 model regresses badly, widen the representative set
(`--max-samples`) or fall back to float16 in `convert_cnn_to_tflite.py`.

## On-hardware validation checklist

- [ ] `model_data.h` generated and int8-vs-float accuracy checked on the test split
- [ ] `idf.py build` clean; confirm the op set in `main.cpp` matches what onnx2tf emitted (GlobalAveragePool → `Mean` **or** `AveragePool2D`)
- [ ] `AllocateTensors` succeeds; note real `arena used` and shrink `kArenaBytes` toward it
- [ ] Latency recorded (feeds Figure 6 / the Phase-2 hardware table)
- [ ] Feed a real captured cell (not the test pattern) and confirm the prediction matches the Pi's for the same cell

[esp-tflite-micro]: https://github.com/espressif/esp-tflite-micro
