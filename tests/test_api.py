from fastapi.testclient import TestClient
import os
import sys
from unittest.mock import patch, MagicMock
import numpy as np

# ---------------------------------------------------------------------------
# FIX: Ensure repo root is added to sys.path (so src.app loads correctly)
# ---------------------------------------------------------------------------

CURRENT_DIR = os.path.dirname(__file__)               # Foodalyze/tests
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))  # Foodalyze/
sys.path.insert(0, PROJECT_ROOT)

from src.app import app

client = TestClient(app)

# --- Define the path to your test image ---
TEST_IMAGE_PATH = os.path.join(CURRENT_DIR, "sample_food.jpg")


# ---------------------------------------------------------------------------
# BASE TESTS
# ---------------------------------------------------------------------------


def test_health_check():
    """Tests if the /health endpoint is working."""
    response = client.get("/health")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "OK"
    assert "model_loaded" in json_data


def test_root_endpoint():
    """Tests the main / endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "name" in response.json()


def test_model_info_endpoint():
    """Tests the /model_info endpoint."""
    response = client.get("/model_info")
    assert response.status_code == 200
    json_data = response.json()
    assert "num_classes" in json_data
    assert "classes" in json_data
    assert isinstance(json_data["classes"], dict)


def test_predict_endpoint():
    """Tests the /predict endpoint with a real image file (if available)."""
    if not os.path.exists(TEST_IMAGE_PATH):
        import pytest
        pytest.skip("sample_food.jpg not found, skipping real prediction test")

    with open(TEST_IMAGE_PATH, "rb") as f:
        response = client.post(
            "/predict",
            files={"file": ("sample_food.jpg", f, "image/jpeg")},
        )

    assert response.status_code == 200
    data = response.json()
    assert "detections" in data


def test_predict_no_file():
    """Tests sending a request to /predict without a file."""
    response = client.post("/predict")
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# HIGHER COVERAGE TESTS — FIXED PATCH PATHS
# ---------------------------------------------------------------------------
# IMPORTANT:
# Your app is at src/app.py → Python module = src.app
# Therefore patch targets must be:
#   "src.app.model"
#   "src.app.cv2.imdecode"
# ---------------------------------------------------------------------------


@patch("src.app.cv2.imdecode", return_value=np.zeros((100, 100, 3), dtype=np.uint8))
@patch("src.app.model")
def test_predict_with_mocked_yolo(mock_model, mock_imdecode):
    """Covers YOLO prediction and calorie lookup logic."""
    mock_box = MagicMock()
    mock_box.xyxy = [[100, 100, 200, 200]]
    mock_box.conf = [0.95]
    mock_box.cls = [0]

    mock_result = MagicMock()
    mock_result.boxes = [mock_box]

    mock_model.predict.return_value = [mock_result]

    response = client.post(
        "/predict",
        files={"file": ("fake.jpg", b"fakebytes", "image/jpeg")},
    )

    assert response.status_code == 200
    data = response.json()
    assert "detections" in data


@patch("src.app.cv2.imdecode", side_effect=Exception("Decode failed"))
def test_predict_error_branch(mock_imdecode):
    """Covers the exception-handling block in /predict."""
    response = client.post(
        "/predict",
        files={"file": ("fake.jpg", b"data", "image/jpeg")},
    )
    assert response.status_code == 400
    assert "Prediction failed" in response.text


def test_predict_model_not_loaded():
    """Covers the 'model is None' branch."""
    import src.app as app_module

    original_model = app_module.model
    app_module.model = None

    try:
        response = client.post(
            "/predict",
            files={"file": ("fake.jpg", b"data", "image/jpeg")},
        )
        assert response.status_code == 500
        assert "Model not loaded" in response.text
    finally:
        app_module.model = original_model
