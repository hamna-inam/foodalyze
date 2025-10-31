from fastapi.testclient import TestClient
import os
import sys
from unittest.mock import patch, MagicMock
import numpy as np

# Ensure project root (Foodalyze) is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from app import app

client = TestClient(app)

# --- Define the path to your test image ---
TEST_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "sample_food.jpg")


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
        # Skip this test if no real image is available
        import pytest

        pytest.skip("sample_food.jpg not found, skipping real prediction test")

    with open(TEST_IMAGE_PATH, "rb") as f:
        response = client.post(
            "/predict", files={"file": ("sample_food.jpg", f, "image/jpeg")}
        )

    assert response.status_code == 200
    json_data = response.json()
    assert "image" in json_data
    assert "num_detections" in json_data
    assert "detections" in json_data
    assert isinstance(json_data["detections"], list)
    if json_data["num_detections"] > 0:
        det = json_data["detections"][0]
        assert "class_name" in det
        assert "confidence" in det
        assert "bbox" in det
        assert "calories_estimate" in det


def test_predict_no_file():
    """Tests sending a request to /predict without a file."""
    response = client.post("/predict")
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# ADDITIONAL TESTS FOR HIGHER COVERAGE (MOCKING YOLO + ERROR HANDLING)
# ---------------------------------------------------------------------------


@patch("app.cv2.imdecode", return_value=np.zeros((100, 100, 3), dtype=np.uint8))
@patch("app.model")
def test_predict_with_mocked_yolo(mock_model, mock_imdecode):
    """Covers YOLO prediction and calorie lookup logic."""
    # Fake YOLO detection
    mock_box = MagicMock()
    mock_box.xyxy = [[100, 100, 200, 200]]
    mock_box.conf = [0.95]
    mock_box.cls = [0]
    mock_result = MagicMock()
    mock_result.boxes = [mock_box]
    mock_model.predict.return_value = [mock_result]

    response = client.post(
        "/predict", files={"file": ("fake.jpg", b"fakebytes", "image/jpeg")}
    )

    assert response.status_code == 200
    data = response.json()
    assert "detections" in data
    assert isinstance(data["detections"], list)
    if data["detections"]:
        det = data["detections"][0]
        assert "confidence" in det
        assert "calories_estimate" in det


@patch("app.cv2.imdecode", side_effect=Exception("Decode failed"))
def test_predict_error_branch(mock_imdecode):
    """Covers the exception-handling block in /predict."""
    response = client.post(
        "/predict", files={"file": ("fake.jpg", b"data", "image/jpeg")}
    )
    assert response.status_code == 400
    assert "Prediction failed" in response.text


def test_predict_model_not_loaded():
    """Covers the 'model is None' branch at start of /predict."""
    import app as app_module

    # Temporarily set model to None
    original_model = app_module.model
    app_module.model = None

    try:
        response = client.post(
            "/predict", files={"file": ("fake.jpg", b"data", "image/jpeg")}
        )
        assert response.status_code == 500
        assert "Model not loaded" in response.text
    finally:
        # Restore model after test
        app_module.model = original_model
