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


def slugify(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


def parse_args():
    p = argparse.ArgumentParser(description="Hierarchical inference: coarse model + specialized model.")
    p.add_argument("--coarse_model", type=str, default="model.tflite")
    p.add_argument("--coarse_labels", type=str, default="labels.txt")
    p.add_argument("--specialized_root", type=str, default="models/specialized")
    p.add_argument("--image", type=str, default=None)
    p.add_argument("--camera", action="store_true")
    p.add_argument("--camera_index", type=int, default=0)
    p.add_argument("--coarse_threshold", type=float, default=0.30)
    p.add_argument("--specialized_threshold", type=float, default=0.20)
    return p.parse_args()


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


def run_on_frame(frame, coarse_interpreter, coarse_labels, specialized_models, coarse_threshold, specialized_threshold):
    coarse_class, coarse_conf = predict(coarse_interpreter, frame, coarse_labels)
    result = {
        "coarse_class": coarse_class,
        "coarse_conf": coarse_conf,
        "specialized_class": None,
        "specialized_conf": None,
    }

    if coarse_conf < coarse_threshold:
        return result

    key = slugify(coarse_class)
    if key not in specialized_models:
        return result

    spec_interpreter = specialized_models[key]["interpreter"]
    spec_labels = specialized_models[key]["labels"]
    spec_class, spec_conf = predict(spec_interpreter, frame, spec_labels)
    if spec_conf >= specialized_threshold:
        result["specialized_class"] = spec_class
        result["specialized_conf"] = spec_conf
    return result


def run_image_mode(args, coarse_interpreter, coarse_labels, specialized_models):
    frame = cv2.imread(args.image)
    if frame is None:
        raise ValueError(f"Could not read image: {args.image}")

    out = run_on_frame(
        frame,
        coarse_interpreter,
        coarse_labels,
        specialized_models,
        args.coarse_threshold,
        args.specialized_threshold,
    )

    print(f"Coarse: {out['coarse_class']} ({out['coarse_conf']:.4f})")
    if out["specialized_class"] is not None:
        print(f"Specialized: {out['specialized_class']} ({out['specialized_conf']:.4f})")
    else:
        print("Specialized: not available / below threshold")


def run_camera_mode(args, coarse_interpreter, coarse_labels, specialized_models):
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {args.camera_index}.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            out = run_on_frame(
                frame,
                coarse_interpreter,
                coarse_labels,
                specialized_models,
                args.coarse_threshold,
                args.specialized_threshold,
            )

            line1 = f"Coarse: {out['coarse_class']} ({out['coarse_conf']:.2f})"
            if out["specialized_class"] is None:
                line2 = "Specialized: n/a"
            else:
                line2 = f"Specialized: {out['specialized_class']} ({out['specialized_conf']:.2f})"

            cv2.putText(frame, line1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, line2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.imshow("Hierarchical Inference", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    args = parse_args()

    coarse_interpreter = load_interpreter(Path(args.coarse_model))
    coarse_labels = load_labels(Path(args.coarse_labels))
    specialized_models = discover_specialized_models(Path(args.specialized_root))

    print(f"Loaded coarse model: {args.coarse_model}")
    print(f"Loaded specialized models: {sorted(specialized_models.keys())}")

    if args.camera:
        run_camera_mode(args, coarse_interpreter, coarse_labels, specialized_models)
    elif args.image:
        run_image_mode(args, coarse_interpreter, coarse_labels, specialized_models)
    else:
        raise ValueError("Provide either --camera or --image.")


if __name__ == "__main__":
    main()
