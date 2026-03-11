# RPi Deploy – Hierarchical Vision Pipeline

This folder contains everything needed to run the real‑time vision pipeline on a Raspberry Pi 4 using TFLite.

## Contents

- `app_rpi_web.py` – main runtime (camera, inference, web UI, optional UART).
- `infer.py` – TFLite inference utilities.
- `infer_hierarchical.py` – CLI hierarchical inference.
- `router.py` – CLI router for specialized models.
- `model.tflite`, `labels.txt` – coarse classifier.
- `models/specialized/*/model.tflite` – specialized classifiers.
- `templates/index.html` – web UI.

## Setup (RPi)

```bash
cd rpi_deploy
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run (no UART, timer via UI button)

```bash
python app_rpi_web.py \
  --coarse_model model.tflite \
  --coarse_labels labels.txt \
  --specialized_root models/specialized \
  --camera_index 0 \
  --enable_qr \
  --enable_ocr \
  --plate_detector_model license_plate_detector.pt \
  --host 0.0.0.0 \
  --port 8080
```

Open: `http://<rpi-ip>:8080`

## Run (with UART)

```bash
python app_rpi_web.py \
  --coarse_model model.tflite \
  --coarse_labels labels.txt \
  --specialized_root models/specialized \
  --camera_index 0 \
  --enable_qr \
  --enable_ocr \
  --plate_detector_model license_plate_detector.pt \
  --uart_port /dev/serial0 \
  --uart_baud 115200 \
  --detect_window_sec 5 \
  --host 0.0.0.0 \
  --port 8080
```

UART messages from Arduino:

- `START` → starts the run timer in the UI.
- `SERVO` → enables detection for `--detect_window_sec` seconds.

## Notes

- If you use full TensorFlow, it will work but is heavier than `tflite-runtime`.
- Plate OCR uses Tesseract. Set `TESSERACT_CMD` if needed.
- The plate detector is optional; without it, OCR runs on the full frame.
