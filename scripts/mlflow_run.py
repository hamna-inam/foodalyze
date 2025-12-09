# mlflow_yolo_params.py

from ultralytics import YOLO
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
import os

# --- Config ---
MODEL_PATH = "best.pt"
MLFLOW_URI = "file:///Users/bstar/Documents/Fall25/MLOps/MLFlow/mlruns"  # change to your MLflow URI
EXPERIMENT_NAME = "YOLOv8_Parameter_Tracking"
RUN_NAME = "best.pt_params"
ARTIFACT_DIR = "mlflow_artifacts"

os.makedirs(ARTIFACT_DIR, exist_ok=True)

# --- Load YOLO model ---
model = YOLO(MODEL_PATH)
pt_model = model.model  # PyTorch model

# --- Extract parameter stats ---
param_stats = {}
for name, param in pt_model.named_parameters():
    if param.requires_grad:
        param_stats[name] = {
            "mean": param.data.mean().item(),
            "std": param.data.std().item(),
            "min": param.data.min().item(),
            "max": param.data.max().item(),
        }

# --- Start MLflow run ---
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name=RUN_NAME) as run:
    # Log parameter statistics as metrics
    for layer, stats in param_stats.items():
        for stat_name, value in stats.items():
            mlflow.log_metric(f"{layer}_{stat_name}", value)

    # Log layer-wise histograms as artifacts
    for name, param in pt_model.named_parameters():
        if param.requires_grad:
            plt.figure(figsize=(8, 4))
            plt.hist(param.data.cpu().numpy().flatten(), bins=50)
            plt.title(f"{name} weights distribution")
            plt.xlabel("Weight value")
            plt.ylabel("Frequency")
            hist_path = os.path.join(ARTIFACT_DIR, f"{name.replace('.', '_')}_hist.png")
            plt.savefig(hist_path)
            plt.close()
            mlflow.log_artifact(hist_path)

    # Log the full PyTorch model
    mlflow.pytorch.log_model(pt_model, artifact_path="yolov8_model")

print(f"✅ Parameter metrics and model logged to MLflow run: {run.info.run_id}")
