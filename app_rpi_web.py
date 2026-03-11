import argparse
import json
import re
import threading
import time
from pathlib import Path
import os
from collections import deque

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template

from infer import load_labels, predict

try:
    import zxingcpp
except ImportError:
    zxingcpp = None

try:
    import pytesseract
except ImportError:
    pytesseract = None
    TESSERACT_AVAILABLE = False
else:
    TESSERACT_AVAILABLE = True

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    try:
        from tensorflow.lite import Interpreter
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    import serial
except Exception:
    serial = None


def slugify(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


def load_interpreter(model_path: Path):
    interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    return interpreter


def discover_specialized_models(specialized_root: Path):
    models = {}
    if not specialized_root.exists():
        return models
    for d in specialized_root.iterdir():
        if not d.is_dir():
            continue
        model_path = d / "model.tflite"
        labels_path = d / "labels.txt"
        if model_path.exists() and labels_path.exists():
            models[d.name] = {
                "interpreter": load_interpreter(model_path),
                "labels": load_labels(labels_path),
            }
    return models


def decode_qr_texts(frame_bgr):
    texts = []
    if zxingcpp is not None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        codes = zxingcpp.read_barcodes(rgb)
        for c in codes:
            if c.text:
                texts.append(c.text)
        return texts

    detector = cv2.QRCodeDetector()
    ok, decoded_info, _, _ = detector.detectAndDecodeMulti(frame_bgr)
    if ok and decoded_info:
        texts.extend([t for t in decoded_info if t])
    return texts


def run_ocr(frame_bgr, config="--psm 7"):
    global TESSERACT_AVAILABLE
    if pytesseract is None or not TESSERACT_AVAILABLE:
        return ""
    tesseract_cmd = os.environ.get("TESSERACT_CMD")
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    try:
        return pytesseract.image_to_string(rgb, config=config).strip()
    except Exception:
        TESSERACT_AVAILABLE = False
        return ""


def preprocess_ocr(gray, args, kind):
    work = gray
    if args.ocr_upscale != 1.0:
        work = cv2.resize(work, None, fx=args.ocr_upscale, fy=args.ocr_upscale, interpolation=cv2.INTER_CUBIC)
    if args.ocr_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        work = clahe.apply(work)
    if args.ocr_blur:
        work = cv2.bilateralFilter(work, 7, 75, 75)

    if args.ocr_thresh == "adaptive":
        work = cv2.adaptiveThreshold(work, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 31, 2)
    elif args.ocr_thresh == "otsu":
        work = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    if args.ocr_invert == "true":
        work = cv2.bitwise_not(work)
    elif args.ocr_invert == "auto":
        if np.mean(work) < 110:
            work = cv2.bitwise_not(work)

    if kind == "plate":
        kernel = np.ones((2, 2), np.uint8)
        work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, kernel, iterations=1)

    return work


def find_plate_roi(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(gray, 30, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    h, w = frame_bgr.shape[:2]
    best = None
    best_area = 0
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        x, y, cw, ch = cv2.boundingRect(approx)
        if ch == 0:
            continue
        aspect = cw / float(ch)
        area = cw * ch
        if area < 0.01 * (w * h):
            continue
        if 2.0 <= aspect <= 6.5 and area > best_area:
            best_area = area
            best = (x, y, cw, ch)
    if best is None:
        return None
    x, y, cw, ch = best
    return frame_bgr[y:y + ch, x:x + cw]


def extract_plate_text(frame_bgr, args):
    roi = find_plate_roi(frame_bgr)
    if roi is None:
        return ""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    proc = preprocess_ocr(gray, args, "plate")
    cfg = f"--psm {args.ocr_plate_psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    text = run_ocr(proc, config=cfg)
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def detect_plate_yolo(frame_bgr, model, conf):
    if model is None:
        return None
    try:
        results = model.predict(frame_bgr, conf=conf, verbose=False)
    except Exception:
        return None
    if not results:
        return None
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return None
    boxes = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    best_idx = int(np.argmax(confs))
    x1, y1, x2, y2 = boxes[best_idx].astype(int).tolist()
    return x1, y1, x2, y2, float(confs[best_idx])


class VideoEngine:
    def __init__(self, args):
        self.args = args
        self.coarse_labels = load_labels(Path(args.coarse_labels))
        self.coarse_interpreter = load_interpreter(Path(args.coarse_model))
        self.specialized_models = discover_specialized_models(Path(args.specialized_root))
        self.plate_detector = None
        if args.plate_detector_model:
            if YOLO is None:
                raise ImportError("ultralytics is required for --plate_detector_model")
            self.plate_detector = YOLO(args.plate_detector_model)

        self.cap = cv2.VideoCapture(args.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        self.cap.set(cv2.CAP_PROP_FPS, args.camera_fps)

        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open camera index {args.camera_index}")

        self._lock = threading.Lock()
        self._latest_frame = None
        self._latest_jpeg = None
        self._latest_meta = {
            "coarse_class": None,
            "coarse_conf": 0.0,
            "specialized_class": None,
            "specialized_conf": 0.0,
            "qr_texts": [],
            "plate_text": "",
            "ocr_enabled": False,
            "tesseract_available": TESSERACT_AVAILABLE,
            "run_timer_sec": 0.0,
            "detect_active": True,
            "detect_window_left": 0.0,
            "fps": 0.0,
            "timestamp": 0.0,
        }
        self._running = True
        self._frame_id = 0
        self._run_start_ts = None
        self._detect_window_end = None
        self._plate_queue = deque(maxlen=2)
        self._plate_last_text = ""
        self._plate_last_ts = 0.0
        self._plate_lock = threading.Lock()

        if args.uart_port:
            if serial is None:
                raise ImportError("pyserial is required for UART. Install with: pip install pyserial")
            self._uart_thread = threading.Thread(target=self._uart_loop, daemon=True)
            self._uart_thread.start()
        if args.enable_ocr:
            self._plate_thread = threading.Thread(target=self._plate_worker, daemon=True)
            self._plate_thread.start()

    def start(self):
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._infer_thread = threading.Thread(target=self._infer_loop, daemon=True)
        self._capture_thread.start()
        self._infer_thread.start()

    def stop(self):
        self._running = False
        self.cap.release()

    def _uart_loop(self):
        try:
            ser = serial.Serial(self.args.uart_port, self.args.uart_baud, timeout=0.5)
        except Exception as exc:
            print(f"UART open failed ({self.args.uart_port}): {exc}")
            return
        print(f"UART connected: {self.args.uart_port} @ {self.args.uart_baud}")
        while self._running:
            try:
                line = ser.readline().decode(errors="ignore").strip()
            except Exception:
                continue
            if not line:
                continue
            msg = line.upper()
            now = time.time()
            if msg.startswith("START"):
                self._run_start_ts = now
            elif msg.startswith("SERVO"):
                self._detect_window_end = now + self.args.detect_window_sec
        try:
            ser.close()
        except Exception:
            pass

    def _capture_loop(self):
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                self._latest_frame = frame

    def _plate_worker(self):
        while self._running:
            if not self._plate_queue:
                time.sleep(0.01)
                continue
            frame = self._plate_queue.popleft()
            plate_text = ""
            if self.plate_detector is not None:
                det = detect_plate_yolo(frame, self.plate_detector, self.args.plate_detector_conf)
                if det is not None:
                    x1, y1, x2, y2, _ = det
                    crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                    if crop.size > 0:
                        plate_text = extract_plate_text(crop, self.args)
            if not plate_text:
                plate_text = extract_plate_text(frame, self.args)
            if plate_text:
                with self._plate_lock:
                    self._plate_last_text = plate_text
                    self._plate_last_ts = time.time()

    def _infer_loop(self):
        last_infer = time.time()
        while self._running:
            with self._lock:
                frame = None if self._latest_frame is None else self._latest_frame.copy()

            if frame is None:
                time.sleep(0.005)
                continue

            self._frame_id += 1
            run_infer = (self._frame_id % self.args.infer_every_n) == 0

            if run_infer:
                now = time.time()
                detect_active = True
                detect_left = 0.0
                if self.args.detect_window_sec > 0:
                    if self._detect_window_end is None:
                        detect_active = False
                    else:
                        detect_left = max(0.0, self._detect_window_end - now)
                        detect_active = detect_left > 0.0

                if not detect_active:
                    self._latest_meta = {
                        "coarse_class": None,
                        "coarse_conf": 0.0,
                        "specialized_class": None,
                        "specialized_conf": 0.0,
                        "qr_texts": [],
                        "plate_text": "",
                        "ocr_enabled": bool(self.args.enable_ocr),
                        "tesseract_available": bool(TESSERACT_AVAILABLE),
                        "run_timer_sec": round((now - self._run_start_ts), 1) if self._run_start_ts else 0.0,
                        "detect_active": False,
                        "detect_window_left": round(detect_left, 2),
                        "fps": 0.0,
                        "timestamp": now,
                    }
                    time.sleep(0.001)
                    continue

                coarse_class, coarse_conf = predict(self.coarse_interpreter, frame, self.coarse_labels)
                specialized_class, specialized_conf = None, 0.0

                if coarse_conf >= self.args.coarse_threshold:
                    key = slugify(coarse_class)
                    if key in self.specialized_models:
                        spec = self.specialized_models[key]
                        s_class, s_conf = predict(spec["interpreter"], frame, spec["labels"])
                        if s_conf >= self.args.specialized_threshold:
                            specialized_class, specialized_conf = s_class, s_conf

                qr_texts = decode_qr_texts(frame) if self.args.enable_qr else []

                plate_text = ""
                if self.args.enable_ocr:
                    coarse_key = slugify(coarse_class)
                    if coarse_key in {"vehicle_number_plates", "vehicle_number_plate", "vehicles"}:
                        if (self._frame_id % self.args.plate_detect_every_n) == 0:
                            if len(self._plate_queue) < self._plate_queue.maxlen:
                                self._plate_queue.append(frame.copy())
                        with self._plate_lock:
                            plate_text = self._plate_last_text

                fps = 1.0 / max(now - last_infer, 1e-6)
                last_infer = now

                gated_coarse_class = coarse_class
                gated_coarse_conf = float(coarse_conf)
                gated_spec_class = specialized_class
                gated_spec_conf = float(specialized_conf)
                gated_plate_text = plate_text

                self._latest_meta = {
                    "coarse_class": gated_coarse_class,
                    "coarse_conf": round(float(gated_coarse_conf), 4),
                    "specialized_class": gated_spec_class,
                    "specialized_conf": round(float(gated_spec_conf), 4),
                    "qr_texts": qr_texts,
                    "plate_text": gated_plate_text,
                    "ocr_enabled": bool(self.args.enable_ocr),
                    "tesseract_available": bool(TESSERACT_AVAILABLE),
                    "run_timer_sec": round((now - self._run_start_ts), 1) if self._run_start_ts else 0.0,
                    "detect_active": True,
                    "detect_window_left": round(detect_left, 2),
                    "fps": round(float(fps), 2),
                    "timestamp": now,
                }

            vis = frame.copy()
            if self.args.stream_scale != 1.0:
                vis = cv2.resize(vis, None, fx=self.args.stream_scale, fy=self.args.stream_scale)
            ok, jpg = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, self.args.jpeg_quality])
            if ok:
                with self._lock:
                    self._latest_jpeg = jpg.tobytes()

            time.sleep(0.001)

    def _draw_overlay(self, frame, meta):
        gate_note = ""
        if self.args.challenge_gate and not meta.get("challenge_gate", False):
            gate_note = "Waiting for Challenge Image"

        c_txt = f"Coarse: {meta['coarse_class']} ({meta['coarse_conf']:.2f})"
        if meta["coarse_class"] is None:
            c_txt = "Coarse: n/a"

        s_txt = "Specialized: n/a" if meta["specialized_class"] is None else (
            f"Specialized: {meta['specialized_class']} ({meta['specialized_conf']:.2f})"
        )
        q_txt = "QR: " + (" | ".join(meta["qr_texts"]) if meta["qr_texts"] else "none")
        plate_txt = f"Plate: {meta.get('plate_text') or 'n/a'}"
        f_txt = f"Infer FPS: {meta['fps']:.1f}"

        cv2.rectangle(frame, (6, 6), (520, 166), (245, 245, 245), thickness=-1)
        cv2.rectangle(frame, (6, 6), (520, 166), (185, 185, 185), thickness=1)
        cv2.putText(frame, c_txt, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (35, 35, 35), 2)
        cv2.putText(frame, s_txt, (14, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (60, 50, 40), 2)
        cv2.putText(frame, q_txt, (14, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (70, 60, 50), 2)
        cv2.putText(frame, plate_txt, (14, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (70, 60, 50), 2)
        cv2.putText(frame, f_txt, (14, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (90, 90, 90), 2)
        if gate_note:
            cv2.putText(frame, gate_note, (14, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (120, 90, 70), 2)
        return frame

    def get_jpeg(self):
        with self._lock:
            return self._latest_jpeg

    def get_meta(self):
        with self._lock:
            return dict(self._latest_meta)


def parse_args():
    p = argparse.ArgumentParser(description="RPi web streaming + hierarchical object detection.")
    p.add_argument("--coarse_model", type=str, default="model.tflite")
    p.add_argument("--coarse_labels", type=str, default="labels.txt")
    p.add_argument("--specialized_root", type=str, default="models/specialized")
    p.add_argument("--camera_index", type=int, default=0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--camera_fps", type=int, default=30)
    p.add_argument("--infer_every_n", type=int, default=2, help="Run inference every N frames.")
    p.add_argument("--coarse_threshold", type=float, default=0.35)
    p.add_argument("--specialized_threshold", type=float, default=0.30)
    p.add_argument("--enable_qr", action="store_true")
    p.add_argument("--enable_ocr", action="store_true", help="Enable OCR for vehicle plate reading.")
    p.add_argument("--plate_detector_model", type=str, default=None, help="Path to YOLOv8 plate detector (.pt or .tflite).")
    p.add_argument("--plate_detector_conf", type=float, default=0.25)
    p.add_argument("--plate_detect_every_n", type=int, default=5, help="Run plate detector every N frames.")
    p.add_argument("--uart_port", type=str, default=None, help="UART port (e.g., /dev/serial0 or COM3).")
    p.add_argument("--uart_baud", type=int, default=115200)
    p.add_argument("--detect_window_sec", type=float, default=5.0, help="Detection window after SERVO signal. Set 0 to disable gating.")
    p.add_argument("--ocr_plate_psm", type=int, default=7)
    p.add_argument("--ocr_upscale", type=float, default=2.0)
    p.add_argument("--ocr_thresh", type=str, default="adaptive", choices=["adaptive", "otsu", "none"])
    p.add_argument("--ocr_invert", type=str, default="auto", choices=["auto", "true", "false"])
    p.add_argument("--ocr_clahe", action="store_true")
    p.add_argument("--ocr_blur", action="store_true")
    p.add_argument("--stream_scale", type=float, default=1.0, help="0.5 for lower bandwidth.")
    p.add_argument("--jpeg_quality", type=int, default=70)
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    return p.parse_args()


def create_app(engine: VideoEngine):
    app = Flask(__name__, template_folder="templates")

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/stream.mjpg")
    def stream():
        def gen():
            while True:
                frame = engine.get_jpeg()
                if frame is None:
                    time.sleep(0.01)
                    continue
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/api/status")
    def status():
        return jsonify(engine.get_meta())

    return app


def main():
    args = parse_args()
    engine = VideoEngine(args)
    engine.start()

    app = create_app(engine)
    print(f"Web UI: http://{args.host}:{args.port}")
    print("Loaded specialized models:", sorted(engine.specialized_models.keys()))
    print("QR decoder:", "zxingcpp" if zxingcpp is not None else "OpenCV fallback")
    try:
        app.run(host=args.host, port=args.port, threaded=True, debug=False)
    finally:
        engine.stop()


if __name__ == "__main__":
    main()
