from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import os
import sys

# --- Ensure project root is in path ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

# --- Import the FastAPI app ---
from src.app import app

# --- Create test client ---
client = TestClient(app)

# ---------------------------------------------------------------------------
# BASE TESTS
# ---------------------------------------------------------------------------

def test_health_check():
    """Tests if the /health endpoint is working."""
    response = client.get("/health")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "OK"

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "name" in response.json()

def test_model_info_endpoint():
    response = client.get("/model_info")
    assert response.status_code == 200
    assert "classes" in response.json()

# ---------------------------------------------------------------------------
# MOCKED PREDICTION TESTS
# ---------------------------------------------------------------------------

@patch("src.app.resources")
def test_predict_endpoint(mock_resources):
    import base64
    # Valid minimal JPEG
    VALID_JPEG_BYTES = base64.b64decode(
        b"/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////2wBDAf//////////////////////////////////////////////////////////////////////////////////////wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAf/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIQAxAAAAH/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAEDAQE/Af/EABQRAQAAAAAAAAAAAAAAAAAAAAH/2gAIAQIBAT8B/9k="
    )

    # Fake YOLO detection result
    fake_box = MagicMock()
    fake_box.xyxy = [[100, 100, 200, 200]]
    fake_box.conf = [0.9]
    fake_box.cls = [0]
    
    fake_result = MagicMock()
    fake_result.boxes = [fake_box]

    mock_model = MagicMock()
    mock_model.predict.return_value = [fake_result]

    # Mock resources to return our fake model
    mock_resources.get.side_effect = lambda key, default=None: {
        "yolo_model": mock_model,
        "id_to_class": {0: "aloo_gobi"},
        "vector_db": None,
        "llm_model": None,
        "llm_tokenizer": None,
    }.get(key, default)

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
    # FastAPI returns 422 for missing required fields
    assert response.status_code == 422 

# --- FIXED ERROR BRANCH TEST ---
@patch("src.app.resources")  # <--- WE MUST MOCK RESOURCES HERE TOO
@patch("src.app.cv2.imdecode", side_effect=Exception("Decode failed"))
def test_predict_error_branch(mock_imdecode, mock_resources):
    # Setup: Ensure YOLO model appears "loaded" so we bypass the 500 error check
    mock_resources.get.side_effect = lambda key, default=None: {
        "yolo_model": MagicMock(), # Model is present
        "id_to_class": {}
    }.get(key, default)

    response = client.post(
        "/predict",
        files={"file": ("fake.jpg", b"data", "image/jpeg")},
    )
    
    # Now it should hit the imdecode exception and return 400
    assert response.status_code == 400
    assert "Prediction failed" in response.text

@patch("src.app.resources")
def test_predict_model_not_loaded(mock_resources):
    # Setup: Explicitly return None for yolo_model
    mock_resources.get.return_value = None

    response = client.post(
        "/predict",
        files={"file": ("fake.jpg", b"data", "image/jpeg")},
    )
    # Should hit the "YOLO model not loaded" check
    assert response.status_code == 500
    assert "YOLO model not loaded" in response.text
