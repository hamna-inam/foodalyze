from fastapi.testclient import TestClient
import os

# This imports the 'app' object from your 'app.py' file
try:
    from app import app
except ImportError:
    # Fallback in case the user did use the 'src' folder
    from src.app import app

client = TestClient(app)

# --- Define the path to your test image ---
TEST_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "sample_food.jpg")


def test_health_check():
    """Tests if the /health endpoint is working."""
    response = client.get("/health")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "OK"
    assert json_data["model_loaded"]


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
    assert len(json_data["classes"]) > 0  # Make sure classes loaded


def test_predict_endpoint():
    """
    Tests the /predict endpoint with a real image file.
    This is the most important test.
    """
    # First, check if the sample image actually exists
    if not os.path.exists(TEST_IMAGE_PATH):
        raise FileNotFoundError(
            f"Test image not found at {TEST_IMAGE_PATH}. "
            f"Please add a 'sample_food.jpg' to your 'tests/' folder."
        )

    # Open the image file and send it as a file upload
    with open(TEST_IMAGE_PATH, "rb") as f:
        response = client.post(
            "/predict", files={"file": ("sample_food.jpg", f, "image/jpeg")}
        )

    # Check for a successful response
    assert response.status_code == 200

    # Check the JSON response structure
    json_data = response.json()
    assert "image" in json_data
    assert "num_detections" in json_data
    assert "detections" in json_data
    assert isinstance(json_data["detections"], list)

    # If a detection was made, check its structure
    if json_data["num_detections"] > 0:
        detection = json_data["detections"][0]
        assert "class_name" in detection
        assert "confidence" in detection
        assert "bbox" in detection
        assert "calories_estimate" in detection


def test_predict_no_file():
    """Tests sending a request to /predict without a file."""
    response = client.post("/predict")
    # 422 Unprocessable Entity is the correct FastAPI error

    assert response.status_code == 422
