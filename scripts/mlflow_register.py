import mlflow
import mlflow.pytorch
from ultralytics import YOLO
from pathlib import Path

# --- Config ---
MODEL_PATH = Path("best.pt")
MLFLOW_TRACKING_URI = "file:///Users/bstar/Documents/Fall25/MLOps/MLFlow/mlruns"
EXPERIMENT_NAME = "YOLOv8_Indian_Food_Detection"
REGISTERED_MODEL_NAME = "Foodalyze_YOLOv8_Detector"

# --- MLflow Setup ---
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# --- Load Model ---
model = YOLO(MODEL_PATH)

# --- Log and Register ---
with mlflow.start_run(run_name="YOLOv8_v1_register") as run:
    # Log model
    mlflow.pytorch.log_model(
        pytorch_model=model.model,
        artifact_path="model",
        registered_model_name=REGISTERED_MODEL_NAME,
    )

    # Log parameters manually
    mlflow.log_param("weights_file", str(MODEL_PATH))
    mlflow.log_param("imgsz", 640)
    mlflow.log_param("epochs", 30)
    mlflow.log_param("device", "cuda")

    print(
        f"Model registered as '{REGISTERED_MODEL_NAME}' in MLflow run {run.info.run_id}"
    )
