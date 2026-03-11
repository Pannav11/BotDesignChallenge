import argparse
from pathlib import Path

import cv2
import numpy as np

from infer import load_labels, predict

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    try:
        from tensorflow.lite import Interpreter
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter


def run_qr_decoder(frame):
    detector = cv2.QRCodeDetector()
    data, points, _ = detector.detectAndDecode(frame)
    if points is not None and data:
        return {"status": "ok", "decoded_text": data}
    return {"status": "no_qr_found"}


def run_face_recognition(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    return {"status": "ok", "num_faces_detected": int(len(faces))}


def run_license_plate_ocr(_frame):
    return {
        "status": "stub",
        "message": "License plate OCR module not implemented yet. Integrate EasyOCR/Tesseract + plate detector.",
    }


def run_pet_breed_classifier(_frame):
    return {
        "status": "stub",
        "message": "Pet breed classifier not implemented yet. Train a second-stage pet-breed model.",
    }


def default_handler(_frame, predicted_class):
    return {
        "status": "noop",
        "message": f"No specialized handler mapped for class '{predicted_class}'.",
    }


def route_to_module(frame, predicted_class):
    if predicted_class == "QR_codes":
        return run_qr_decoder(frame)
    if predicted_class == "Face_recognition":
        return run_face_recognition(frame)
    if predicted_class == "Vehicle_number_plates":
        return run_license_plate_ocr(frame)
    if predicted_class == "Pets":
        return run_pet_breed_classifier(frame)
    return default_handler(frame, predicted_class)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Classifier router: classifies frame then calls specialized module.")
    parser.add_argument("--model", type=str, default="model.tflite")
    parser.add_argument("--labels", type=str, default="labels.txt")
    parser.add_argument("--image", type=str, default=None, help="Path to input image.")
    parser.add_argument("--camera", action="store_true", help="Use camera stream.")
    parser.add_argument("--threshold", type=float, default=0.3, help="Minimum classifier confidence to route.")
    return parser


def run_image_mode(interpreter, labels, image_path, threshold):
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError(f"Failed to read image: {image_path}")

    predicted_class, confidence = predict(interpreter, frame, labels)
    print(f"Classifier -> class: {predicted_class} | confidence: {confidence:.4f}")

    if confidence < threshold:
        print(f"Skipping specialized module (below threshold {threshold:.2f}).")
        return

    result = route_to_module(frame, predicted_class)
    print("Module result:", result)


def run_camera_mode(interpreter, labels, threshold):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to open camera (cv2.VideoCapture(0)).")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            predicted_class, confidence = predict(interpreter, frame, labels)
            if confidence >= threshold:
                result = route_to_module(frame, predicted_class)
                module_status = result.get("status", "na")
            else:
                result = {"status": "low_confidence"}
                module_status = "low_confidence"

            label = f"{predicted_class} {confidence:.2f} | {module_status}"
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Classifier Router", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    args = build_arg_parser().parse_args()
    labels = load_labels(Path(args.labels))
    interpreter = Interpreter(model_path=args.model)
    interpreter.allocate_tensors()

    if args.camera:
        run_camera_mode(interpreter, labels, args.threshold)
        return
    if args.image:
        run_image_mode(interpreter, labels, Path(args.image), args.threshold)
        return
    raise ValueError("Provide either --image <path> or --camera.")


if __name__ == "__main__":
    main()
