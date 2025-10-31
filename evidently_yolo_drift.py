import os
import cv2
import numpy as np
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from fastapi import FastAPI
from fastapi.responses import FileResponse
import uvicorn
import threading

# ----------------------------
# Step 1: Extract image features
# ----------------------------
def extract_image_features(image_dir):
    features = []
    for img_name in os.listdir(image_dir):
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        img_path = os.path.join(image_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        contrast = np.std(gray)
        mean_color = np.mean(img, axis=(0, 1))
        features.append({
            "image": img_name,
            "brightness": brightness,
            "contrast": contrast,
            "mean_r": mean_color[2],
            "mean_g": mean_color[1],
            "mean_b": mean_color[0]
        })
    return pd.DataFrame(features)

# ----------------------------
# Step 2: Paths to train/test image directories
# ----------------------------
train_dir = "/Users/bstar/Documents/Fall25/MLOps/MLFlow/IndianFoodDatasetFinalFiltered/train/images"
test_dir = "/Users/bstar/Documents/Fall25/MLOps/MLFlow/IndianFoodDatasetFinalFiltered/test/images"

reference_df = extract_image_features(train_dir)
current_df = extract_image_features(test_dir)

print(f"Reference (train) features: {reference_df.shape}")
print(f"Current (test) features: {current_df.shape}")

# ----------------------------
# Step 3: Generate Evidently HTML report
# ----------------------------
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference_df, current_data=current_df)
report_file = "yolo_data_drift_report.html"
report.save_html(report_file)
print(f"Drift report saved as '{report_file}'")

# ----------------------------
# Step 4: Serve the report locally
# ----------------------------
app = FastAPI()

@app.get("/")
def read_dashboard():
    return FileResponse(report_file)

def start_server():
    uvicorn.run(app, host="0.0.0.0", port=7001)

# Run server in a separate thread
thread = threading.Thread(target=start_server, daemon=True)
thread.start()

print("Dashboard is running at http://localhost:7001")
input("Press Enter to exit...\n")
