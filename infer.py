import argparse
import time
from pathlib import Path

import cv2
import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    try:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
    except Exception as exc:
        raise ImportError(
            "No usable TFLite interpreter found. Install either 'tflite-runtime' "
            "or full 'tensorflow'."
        ) from exc


IMAGE_SIZE = (224, 224)


def parse_args():
    parser = argparse.ArgumentParser(description="Run TFLite image classification inference.")
    parser.add_argument("--model", type=str, default="model.tflite")
    parser.add_argument("--labels", type=str, default="labels.txt")
    parser.add_argument("--image", type=str, default=None, help="Path to input image.")
    parser.add_argument("--camera", action="store_true", help="Use webcam with OpenCV VideoCapture(0).")
    parser.add_argument("--threshold", type=float, default=0.0, help="Confidence threshold.")
    return parser.parse_args()


def load_labels(labels_path: Path):
    if not labels_path.exists():
        raise FileNotFoundError(f"labels.txt not found: {labels_path}")
    return [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def preprocess_bgr_frame(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, IMAGE_SIZE)
    return resized.astype(np.float32)


def prepare_input(input_data, input_details):
    input_dtype = input_details["dtype"]
    input_shape = input_details["shape"]

    if input_shape[0] == 1:
        input_data = np.expand_dims(input_data, axis=0)

    if input_dtype == np.float32:
        return input_data.astype(np.float32)

    scale, zero_point = input_details["quantization"]
    if scale == 0:
        raise ValueError("Invalid input quantization scale 0.")
    quantized = input_data / scale + zero_point
    if input_dtype == np.uint8:
        quantized = np.clip(quantized, 0, 255)
    return quantized.astype(input_dtype)


def dequantize_output(output_data, output_details):
    out_dtype = output_details["dtype"]
    if out_dtype == np.float32:
        return output_data.astype(np.float32)

    scale, zero_point = output_details["quantization"]
    if scale == 0:
        return output_data.astype(np.float32)
    return (output_data.astype(np.float32) - zero_point) * scale


def predict(interpreter, frame, labels):
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_data = preprocess_bgr_frame(frame)
    input_tensor = prepare_input(input_data, input_details)
    interpreter.set_tensor(input_details["index"], input_tensor)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details["index"])
    output = dequantize_output(output, output_details)
    probs = output[0] if output.ndim == 2 else output

    class_id = int(np.argmax(probs))
    confidence = float(probs[class_id])
    class_name = labels[class_id] if class_id < len(labels) else str(class_id)
    return class_name, confidence


def run_image_mode(interpreter, labels, image_path, threshold):
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError(f"Failed to read image: {image_path}")

    class_name, confidence = predict(interpreter, frame, labels)
    if confidence < threshold:
        print(f"Prediction below threshold ({threshold:.2f}): {class_name} ({confidence:.4f})")
    else:
        print(f"Predicted class: {class_name}")
        print(f"Confidence: {confidence:.4f}")


def run_camera_mode(interpreter, labels, threshold):
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise RuntimeError("Unable to open camera (cv2.VideoCapture(1)).")

    prev_time = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            class_name, confidence = predict(interpreter, frame, labels)
            label = f"{class_name}: {confidence:.2f}"
            if confidence < threshold:
                label = f"low_conf: {class_name} {confidence:.2f}"

            cur_time = time.time()
            fps = 1.0 / max(cur_time - prev_time, 1e-6)
            prev_time = cur_time

            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.imshow("TFLite Classifier", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    args = parse_args()
    labels = load_labels(Path(args.labels))

    interpreter = Interpreter(model_path=args.model)
    interpreter.allocate_tensors()

    if args.camera:
        run_camera_mode(interpreter, labels, args.threshold)
    elif args.image:
        run_image_mode(interpreter, labels, Path(args.image), args.threshold)
    else:
        raise ValueError("Provide either --image <path> or --camera.")


if __name__ == "__main__":
    main()
