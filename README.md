# PillWatch

PillWatch (formerly pillbox) is a student research project: camera capture and computer-vision pill-occupancy detection for a 21-cell pill organizer.

Take photos from any browser using a Raspberry Pi camera, a Linux UVC webcam
(including a Wyze Cam v2 in webcam mode), or an ESP32-CAM over Wi-Fi — live
preview, still capture, and a gallery to browse and download what you've taken.

## Using it

Open **https://pi.uprobotics.tech** on your phone or computer — works from anywhere,
via a Cloudflare Tunnel. (On the home network you can also use `http://<pi-ip>:8000/`
directly.) You may see a brief Cloudflare "Just a moment…" check before the page loads.

The first visit asks for a **4-digit access code**; after that your browser is
remembered for 30 days (and until the app restarts).

**Camera page (`/`)**
- Live preview with a red shutter button — tap it to take a photo.
- Photos are captured at the camera's full 12MP resolution; a brief "Capturing…"
  overlay appears while the shot is taken.

**Gallery (`/gallery`)**
- Every photo you've taken, newest first.
- Download photos one at a time, or everything as a zip.
- Use **Select** to act on many photos at once: **Download selected** (a single
  photo comes down on its own, several arrive as a zip) or **Delete selected**.
- Shows how much space photos use and how much is left on the SD card. If the card
  runs low you'll be warned; when critically full, capture is blocked until you
  delete or download some photos. Nothing is ever deleted automatically.

**Status (`/status`)**
- Shows which of the 21 pillbox cells (7 days × morning/noon/night) contain a
  pill in the latest photo, as a green/grey grid. Append `?photo=<name>` to
  check an older photo.
- Detection runs locally on the device with a small CNN — see [detect/README.md](detect/README.md).
  It needs `opencv-python-headless`, `numpy` and `onnxruntime` installed; the
  page tells you if something is missing.

Photos live on the device in `~/photos`, named by date and time
(e.g. `photo_20260712_123030.jpg`).

## Setup

See [INSTALL.md](INSTALL.md) for Pi 4, Pi 5, Jetson/Wyze, and ESP32-CAM
installation, running it as an auto-starting daemon, and the standalone test
scripts in `utils/`.

The Waveshare ESP32-S3-CAM-OV5640 has its own camera-only firmware in
[`firmware/waveshare_esp32s3_ov5640`](firmware/waveshare_esp32s3_ov5640/README.md).
