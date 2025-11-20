import json
import shutil
from pathlib import Path
import yaml
from ultralytics import YOLO

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "people.yaml"
MODEL_DIR = Path("models/people")
METRIC_FILE = Path("models/best_scores.json")
MODEL_KEY = "people"

def load_config(config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return config

def should_replace(new_score):
    if not METRIC_FILE.exists():
        return True
    data = json.load(METRIC_FILE.open())
    return new_score > data.get(MODEL_KEY, {}).get("score", 0.0)

def record_best(run_dir, score):
    data = {}
    if METRIC_FILE.exists():
        data = json.load(METRIC_FILE.open())
    data[MODEL_KEY] = {"score": score, "run": run_dir.name}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(run_dir/"weights"/"best.pt", MODEL_DIR/f"{MODEL_KEY}.pt")
    json.dump(data, METRIC_FILE.open("w"), indent=2)

def train_people():
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

    test_results = model.val(
        data=str(CONFIG_PATH),
        split="test",
        imgsz=params["imgsz"],
        project=params["project"],
        name=params["name"] + "-test",
        device=params["device"],
        workers=params["workers"],
        save_json=True,
    )
    print("test metrics stored under project/name-test")

    precision, recall, map50, map50_95 = test_results.mean_results()

    if should_replace(map50_95):
        record_best(Path(results.save_dir), map50_95)
    else:
        print(f"Existing best checkpoint is better ({METRIC_FILE}).")

if __name__ == "__main__":
    train_people()