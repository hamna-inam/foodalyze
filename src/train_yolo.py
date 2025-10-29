import mlflow
import mlflow.pyfunc
from ultralytics import YOLO

# Start an MLflow run
mlflow.set_experiment("Foodalyze-YOLOv8")

with mlflow.start_run():
    # Log basic hyperparameters
    mlflow.log_param("model", "yolov8n.pt")
    mlflow.log_param("epochs", 30)
    mlflow.log_param("img_size", 640)

    # Initialize YOLO model
    model = YOLO("yolov8n.pt")

    # Train model
    results = model.train(
        data="data/IndianFoodDatasetFinalFiltered/data.yaml",  # path to your dataset YAML
        epochs=30,
        imgsz=640,
        batch=16,
        project="runs/train",
        name="foodalyze_yolov8",
    )

    # Log results to MLflow
    mlflow.log_metric("mAP50", results.results_dict.get("metrics/mAP50", 0))
    mlflow.log_metric("precision", results.results_dict.get("metrics/precision", 0))
    mlflow.log_metric("recall", results.results_dict.get("metrics/recall", 0))

    # Log model weights
    best_model_path = results.best
    mlflow.log_artifact(best_model_path)

    print("Training complete and logged to MLflow.")
