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
PALETTE = [
    (255, 56, 56),
    (56, 255, 56),
    (56, 56, 255),
    (255, 165, 0),
    (0, 255, 255),
    (255, 0, 255),
]


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


def color_for_idx(idx: int) -> Tuple[int, int, int]:
    return PALETTE[idx % len(PALETTE)]


def draw_boxes(frame, boxes: List[Dict[str, object]], base_color: Optional[Tuple[int, int, int]]) -> None:
    for box in boxes:
        x1, y1, x2, y2 = map(int, (box["x1"], box["y1"], box["x2"], box["y2"]))
        cls_id = int(box["class_id"])
        color = color_for_idx(cls_id) if base_color is None else base_color
        label = f'{box["class_name"]} {box["confidence"]:.2f}'

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_text = max(y1 - 4, th + baseline)
        cv2.rectangle(frame, (x1, y_text - th - baseline), (x1 + tw, y_text + baseline), color, -1)
        cv2.putText(frame, label, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


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


def run_video_with_overlays(
    ball_model: YOLO,
    people_model: YOLO,
    source: Path,
    run_dir: Path,
    conf: float,
    device: Optional[str],
    show: bool,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Path]:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    overlay_path = run_dir / "overlay.mp4"
    writer = cv2.VideoWriter(
        str(overlay_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    ball_dets: List[Dict[str, object]] = []
    people_dets: List[Dict[str, object]] = []
    frame_idx = 0
    window_name = "Combined detections"

    # Preload to device for faster per-frame calls
    if device:
        ball_model.to(device)
        people_model.to(device)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ball_res = ball_model.predict(source=frame, conf=conf, device=device, verbose=False)[0]
        people_res = people_model.predict(source=frame, conf=conf, device=device, verbose=False)[0]

        ball_boxes = extract_boxes(ball_res, ball_model.names)
        people_boxes = extract_boxes(people_res, people_model.names)

        ball_dets.append({"frame": frame_idx, "boxes": ball_boxes})
        people_dets.append({"frame": frame_idx, "boxes": people_boxes})

        annotated = frame.copy()
        draw_boxes(annotated, people_boxes, base_color=None)
        draw_boxes(annotated, ball_boxes, base_color=(0, 255, 255))
        writer.write(annotated)

        if show:
            cv2.imshow(window_name, annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Display interrupted by user")
                break

        frame_idx += 1

    cap.release()
    writer.release()
    if show:
        cv2.destroyWindow(window_name)

    print(f"[combined] frames: {frame_idx}, overlay saved to {overlay_path}")
    return ball_dets, people_dets, overlay_path


def parse_args():
    parser = ArgumentParser(description="Run inference on an image or video with ball and people detectors.")
    parser.add_argument("source", type=Path, help="Path to the input image or video.")
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
    is_video = args.source.suffix.lower() in VIDEO_EXTS

    if is_video:
        ball_dets, people_dets, overlay_path = run_video_with_overlays(
            ball_model, people_model, args.source, run_dir, args.conf, args.device, args.show
        )
        models_payload = {
            "ball": ball_dets,
            "people": people_dets,
        }
        overlay_entry = str(overlay_path)
    else:
        ball_results = run_model(ball_model, args.source, run_dir, "ball", args.conf, args.device, args.show)
        people_results = run_model(people_model, args.source, run_dir, "people", args.conf, args.device, args.show)
        models_payload = {
            "ball": ball_results["detections"],
            "people": people_results["detections"],
        }
        overlay_entry = None

    detections = {
        "source": str(args.source),
        "run_dir": str(run_dir),
        "overlay_video": overlay_entry,
        "models": models_payload,
    }
    json_path = run_dir / "detections.json"
    json_path.write_text(json.dumps(detections, indent=2))
    print(f"Detections JSON written to {json_path}")


if __name__ == "__main__":
    main()
