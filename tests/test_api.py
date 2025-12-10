from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np
import os
import sys

# --- Ensure project root is in path ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

# --- Import the FastAPI app ---
from src.app import app  # noqa

# --- Diagnostics: These DO NOT violate Ruff E402 now ---
print("APP TYPE:", type(app))
print("APP MODULE:", getattr(app, "__module__", None))
print("APP ATTRS SAMPLE:", list(dir(app))[:20])

# --- Create test client ---
client = TestClient(app)


# Path to test image
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
    assert "timestamp" in json_data
    assert "llm_loaded" in json_data
    assert "rag_loaded" in json_data


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


@patch("src.app.resources")
def test_predict_endpoint(mock_resources):
    # ---- Valid minimal JPEG ----
    import base64

    VALID_JPEG_BYTES = base64.b64decode(
        b"/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////2wBDAf//////////////////////////////////////////////////////////////////////////////////////wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAf/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIQAxAAAAH/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAEDAQE/Af/EABQRAQAAAAAAAAAAAAAAAAAAAAH/2gAIAQIBAT8B/9k="
    )

    # ---- Fake YOLO detection ----
    fake_box = MagicMock()
    fake_box.xyxy = [[100, 100, 200, 200]]
    fake_box.conf = [0.9]
    fake_box.cls = [0]

    fake_result = MagicMock()
    fake_result.boxes = [fake_box]

    mock_model = MagicMock()
    mock_model.predict.return_value = [fake_result]

    # ---- Mock resources ----
    mock_resources.get.side_effect = lambda key, default=None: {
        "yolo_model": mock_model,
        "id_to_class": {0: "aloo_gobi"},
        "vector_db": None,
        "llm_model": None,
        "llm_tokenizer": None,
    }.get(key, default)

    # ---- Call API ----
    response = client.post(
        "/predict",
        files={"file": ("sample_food.jpg", VALID_JPEG_BYTES, "image/jpeg")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["num_detections"] == 1
    assert body["detections"][0]["class_name"] == "aloo_gobi"


def test_predict_no_file():
    response = client.post("/predict")
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# MOCKED TESTS (correct module paths!)
# ---------------------------------------------------------------------------


@patch("src.app.resources")
def test_predict_with_mocked_yolo(mock_resources):
    import base64
    VALID_JPEG_BYTES = base64.b64decode(
        b"/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////2wBDAf//////////////////////////////////////////////////////////////////////////////////////wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAf/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIQAxAAAAH/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAEDAQE/Af/EABQRAQAAAAAAAAAAAAAAAAAAAAH/2gAIAQIBAT8B/9k="
    )

    # Fake YOLO box
    fake_box = MagicMock()
    fake_box.xyxy = [[100, 100, 200, 200]]
    fake_box.conf = [0.95]
    fake_box.cls = [0]

    fake_result = MagicMock()
    fake_result.boxes = [fake_box]

    mock_yolo = MagicMock()
    mock_yolo.predict.return_value = [fake_result]

    # Mock resources.get()
    mock_resources.get.side_effect = lambda key, default=None: {
        "yolo_model": mock_yolo,
        "id_to_class": {0: "aloo_gobi"},
        "vector_db": None,
        "llm_model": None,
        "llm_tokenizer": None,
    }.get(key, default)

    response = client.post(
        "/predict",
        files={"file": ("img.jpg", VALID_JPEG_BYTES, "image/jpeg")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["num_detections"] == 1
    assert data["detections"][0]["class_name"] == "aloo_gobi"



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

