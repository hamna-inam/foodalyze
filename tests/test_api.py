from fastapi.testclient import TestClient
import os
import sys
from unittest.mock import patch, MagicMock
import numpy as np
import inspect  # <-- moved to top to satisfy Ruff E402

# Ensure project root is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.app import app  # noqa: E402

# Diagnostic print (ALLOWED, Ruff does not block prints)
print("ACTUALLY IMPORTED APP MODULE:", inspect.getfile(app))  # noqa: E402

client = TestClient(app)


# Path to test image
TEST_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "sample_food.jpg")


# ---------------------------------------------------------------------------
# BASE TESTS
# ---------------------------------------------------------------------------


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "OK"
    assert "model_loaded" in json_data


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "name" in response.json()


def test_model_info_endpoint():
    response = client.get("/model_info")
    assert response.status_code == 200
    json_data = response.json()
    assert "num_classes" in json_data
    assert "classes" in json_data
    assert isinstance(json_data["classes"], dict)


def test_predict_endpoint():
    if not os.path.exists(TEST_IMAGE_PATH):
        import pytest

        pytest.skip("sample_food.jpg not found, skipping")

    with open(TEST_IMAGE_PATH, "rb") as f:
        response = client.post(
            "/predict",
            files={"file": ("sample_food.jpg", f, "image/jpeg")},
        )

    assert response.status_code == 200
    data = response.json()
    assert "detections" in data
    assert "num_detections" in data


def test_predict_no_file():
    response = client.post("/predict")
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# MOCKED TESTS (correct module paths!)
# ---------------------------------------------------------------------------


@patch("src.app.cv2.imdecode", return_value=np.zeros((100, 100, 3), dtype=np.uint8))
@patch("src.app.model")
def test_predict_with_mocked_yolo(mock_model, mock_imdecode):
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
    response = client.post(
        "/predict",
        files={"file": ("fake.jpg", b"data", "image/jpeg")},
    )
    assert response.status_code == 400
    assert "Prediction failed" in response.text


def test_predict_model_not_loaded():
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


