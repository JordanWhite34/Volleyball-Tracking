import json
import shutil
from pathlib import Path
import yaml
from ultralytics import YOLO

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "ball.yaml"

def load_config(config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return config


def train_ball():
    cfg = load_config(CONFIG_PATH)
    params = cfg["training_params"]

    model = YOLO(params["weights"])

    results = model.train(
        data=str(CONFIG_PATH),
        epochs=params["epochs"],
        batch=params["batch"],
        imgsz=params["imgsz"],
        device=params["device"],
        workers=params["workers"],
        optimizer=params["optimizer"],
        lr0=params["lr0"],
        patience=params["patience"],
        seed=params["seed"],
        resume=params["resume"],
        exist_ok=params["exist_ok"],
        project=params["project"],
        name=params["name"],
    )
    print("Training run saved to", results.save_dir)

    if params.get("run_test", False):
        model.val(
            data=str(CONFIG_PATH),
            split="test",
            imgsz=params["imgsz"],
            project=params["project"],
            name=params["name"] + "-test",
            device=params.get("device", "auto"),
            workers=params.get("workers", 8),
        )
        print("test metrics stored under project/name-test")

if __name__ == "__main__":
    train_ball()