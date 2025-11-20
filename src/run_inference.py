import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
BALL_MODEL = ROOT / "models" / "ball" / "ball.pt"
PEOPLE_MODEL = ROOT / "models" / "people" / "people.pt"
RUNS_ROOT = ROOT / "inference"
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


def ensure_exists(path: Path, kind: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{kind} not found: {path}")


def next_run_dir(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    existing = []
    for d in base.iterdir():
        if d.is_dir() and d.name.startswith("run_"):
            try:
                existing.append(int(d.name.split("_", 1)[1]))
            except ValueError:
                continue
    idx = (max(existing) + 1) if existing else 1
    run_dir = base / f"run_{idx}"
    run_dir.mkdir()
    return run_dir


def load_models() -> Tuple[YOLO, YOLO]:
    ensure_exists(BALL_MODEL, "Ball model")
    ensure_exists(PEOPLE_MODEL, "People model")
    return YOLO(str(BALL_MODEL)), YOLO(str(PEOPLE_MODEL))


def extract_boxes(result: "YOLOResults", class_names: Dict[int, str]) -> List[Dict[str, object]]:
    boxes_data: List[Dict[str, object]] = []
    if result.boxes is None:
        return boxes_data
    for box in result.boxes:
        xyxy = box.xyxy[0].tolist()
        cls_id = int(box.cls[0])
        boxes_data.append(
            {
                "x1": float(xyxy[0]),
                "y1": float(xyxy[1]),
                "x2": float(xyxy[2]),
                "y2": float(xyxy[3]),
                "confidence": float(box.conf[0]),
                "class_id": cls_id,
                "class_name": class_names.get(cls_id, str(cls_id)),
            }
        )
    return boxes_data


def run_model(
    model: YOLO,
    source: Path,
    project: Path,
    name: str,
    conf: float,
    device: Optional[str],
    show: bool,
) -> Dict[str, object]:
    is_video = source.suffix.lower() in VIDEO_EXTS
    predict_args = dict(
        source=str(source),
        conf=conf,
        project=str(project),
        name=name,
        exist_ok=True,
        device=device,
        save=True,
    )

    detections: List[Dict[str, object]] = []
    save_dir: Optional[Path] = None
    window_name = f"{name} detections"

    if is_video:
        results_iter = model.predict(stream=True, **predict_args)
        for frame_idx, result in enumerate(results_iter):
            boxes_data = extract_boxes(result, model.names)
            detections.append({"frame": frame_idx, "boxes": boxes_data})
            save_dir = Path(result.save_dir)
            if show:
                frame = result.plot()
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print(f"{name}: display interrupted by user")
                    break
        if show:
            cv2.destroyWindow(window_name)
    else:
        results = model.predict(stream=False, **predict_args)
        for frame_idx, result in enumerate(results):
            boxes_data = extract_boxes(result, model.names)
            detections.append({"frame": frame_idx, "boxes": boxes_data})
            save_dir = Path(result.save_dir)
            if show:
                frame = result.plot()
                cv2.imshow(window_name, frame)
                cv2.waitKey(0)
        if show:
            cv2.destroyWindow(window_name)

    total_boxes = sum(len(frame["boxes"]) for frame in detections)
    output_dir = save_dir if save_dir else project / name
    print(f"[{name}] frames: {len(detections)}, boxes: {total_boxes} (saved to {output_dir})")

    return {"model": name, "save_dir": str(output_dir), "detections": detections}


def parse_args():
    parser = ArgumentParser(description="Run inference on an image or video with ball and people detectors.")
    parser.add_argument("source", type=Path, help="Path to the input image or .mp4 video.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=RUNS_ROOT,
        help="Base directory to store outputs (default: inference/).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections (default: 0.25).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on, e.g. 'cuda:0' or 'cpu' (default: autodetect).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display annotated frames (press 'q' to stop for video).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_exists(args.source, "Input")
    run_dir = next_run_dir(args.output)

    ball_model, people_model = load_models()
    ball_results = run_model(ball_model, args.source, run_dir, "ball", args.conf, args.device, args.show)
    people_results = run_model(people_model, args.source, run_dir, "people", args.conf, args.device, args.show)

    detections = {
        "source": str(args.source),
        "run_dir": str(run_dir),
        "models": {
            "ball": ball_results["detections"],
            "people": people_results["detections"],
        },
    }
    json_path = run_dir / "detections.json"
    json_path.write_text(json.dumps(detections, indent=2))
    print(f"Detections JSON written to {json_path}")


if __name__ == "__main__":
    main()
