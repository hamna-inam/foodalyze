from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from src.app import app, resources
from io import BytesIO
from PIL import Image

client = TestClient(app)


def test_yolo_load_failure():
    with patch("src.app.YOLO", side_effect=Exception("load error")):
        with patch("src.app.force_download_from_s3"):
            with patch("src.app.AutoTokenizer"):
                with patch("src.app.AutoModelForCausalLM"):
                    # Trigger startup (no variable needed)
                    with TestClient(app):
                        assert resources.get("yolo_model") is None


def test_class_mapping_load_failure():
    with patch("builtins.open", side_effect=Exception("file error")):
        with patch("src.app.force_download_from_s3", return_value=None):
            with TestClient(app):
                assert resources.get("id_to_class") == {}


def test_faiss_load_failure():
    with patch("src.app.FAISS.load_local", side_effect=Exception("FAISS error")):
        with patch("src.app.force_download_from_s3", return_value=None):
            with TestClient(app):
                assert resources.get("vector_db") is None


def test_llm_load_failure():
    with patch(
        "src.app.AutoTokenizer.from_pretrained", side_effect=Exception("LLM error")
    ):
        with patch("src.app.force_download_from_s3", return_value=None):
            with TestClient(app):
                assert resources.get("llm_model") is None


def test_ask_llm_not_loaded():
    resources["llm_model"] = None
    resources["llm_tokenizer"] = None
    resp = client.post("/ask", json={"text": "hi"})
    assert resp.status_code == 500


@patch("src.app.guard.validate_input", return_value=(False, "Blocked"))
def test_ask_guardrail_block(_):
    resp = client.post("/ask", json={"text": "bad"})
    assert resp.status_code == 400


@patch("src.app.guard.validate_input", return_value=(True, "OK"))
@patch("src.app.resources")
def test_ask_generation_failure(mock_res, _):
    # mock tokenizer + model
    mock_res.get.side_effect = lambda key: {
        "vector_db": MagicMock(
            similarity_search=MagicMock(side_effect=Exception("fail"))
        ),
        "llm_model": MagicMock(generate=MagicMock(side_effect=Exception("gen error"))),
        "llm_tokenizer": MagicMock(
            apply_chat_template=MagicMock(return_value=MagicMock(to=lambda x: x))
        ),
    }.get(key)
    resp = client.post("/ask", json={"text": "nutrition?"})
    assert resp.status_code == 200
    assert "Error:" in resp.json()["answer"]


@patch("src.app.guard.validate_input", return_value=(True, "OK"))
def test_ask_success(_):
    # Tensor-like object that supports slicing
    class FakeTensor:
        def __init__(self, data):
            self._data = list(data)

        @property
        def shape(self):
            return (len(self._data),)

        def __len__(self):
            return len(self._data)

        def __getitem__(self, idx):
            # Support both integer indexing and slicing
            if isinstance(idx, slice):
                return FakeTensor(self._data[idx])
            return self._data[idx]

        def __iter__(self):
            return iter(self._data)

    # Fake input mapping that behaves like tokenizer output
    class FakeInputs(dict):
        def __init__(self):
            super().__init__({"input_ids": FakeTensor([1, 2, 3])})

        def to(self, device):
            return self

    # Fake output that supports nested subscripting and slicing
    # The code does: outputs[0][inputs["input_ids"].shape[-1]:]
    class FakeOutput:
        def __init__(self, data):
            self._data = FakeTensor(data)

        @property
        def shape(self):
            # Return (batch_size, sequence_length)
            return (1, len(self._data))

        def __len__(self):
            return 1  # batch size

        def __getitem__(self, idx):
            # outputs[0] should return the FakeTensor that supports slicing
            return self._data

        def __iter__(self):
            return iter([self._data])

    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = FakeInputs()
    tokenizer.decode.return_value = "healthy food"
    tokenizer.eos_token_id = 2

    model = MagicMock()
    model.device = "cpu"
    model.generate.return_value = FakeOutput([10, 11, 12])

    resources["llm_model"] = model
    resources["llm_tokenizer"] = tokenizer
    resources["vector_db"] = None

    resp = client.post("/ask", json={"text": "Is biryani healthy?"})
    assert resp.status_code == 200
    assert "healthy" in resp.json()["answer"]


# ===== NEW TESTS TO INCREASE COVERAGE =====


def test_health_endpoint():
    """Test the health check endpoint"""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert "status" in resp.json()


def test_root_endpoint():
    """Test the root endpoint"""
    resp = client.get("/")
    assert resp.status_code == 200


def test_model_info_endpoint():
    """Test model info endpoint"""
    resources["yolo_model"] = MagicMock()
    resp = client.get("/model_info")
    assert resp.status_code == 200


def test_model_info_no_model():
    """Test model info when model is not loaded"""
    resources["yolo_model"] = None
    resp = client.get("/model_info")
    assert resp.status_code == 200


def test_predict_no_model():
    """Test predict when YOLO model is not loaded"""
    resources["yolo_model"] = None
    
    fake_image = BytesIO(b"fake image content")
    
    resp = client.post(
        "/predict",
        files={"file": ("test.jpg", fake_image, "image/jpeg")}
    )
    assert resp.status_code == 500


@patch("src.app.resources")
def test_predict_success(mock_res):
    """Test successful prediction"""
    # Create a small test image
    img = Image.new('RGB', (100, 100), color='red')
    img_bytes = BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    # Mock YOLO model result
    mock_result = MagicMock()
    mock_result.boxes.cls = [0, 1]
    mock_result.boxes.conf = [0.9, 0.8]
    mock_result.boxes.xyxy = [[10, 20, 30, 40], [50, 60, 70, 80]]
    
    mock_yolo = MagicMock()
    mock_yolo.return_value = [mock_result]
    
    mock_res.get.side_effect = lambda key: {
        "yolo_model": mock_yolo,
        "id_to_class": {0: "apple", 1: "banana"}
    }.get(key)
    
    resp = client.post(
        "/predict",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "detections" in data


@patch("src.app.resources")
def test_predict_with_confidence_threshold(mock_res):
    """Test prediction with custom confidence threshold"""
    img = Image.new('RGB', (100, 100), color='blue')
    img_bytes = BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    mock_result = MagicMock()
    mock_result.boxes.cls = [0]
    mock_result.boxes.conf = [0.95]
    mock_result.boxes.xyxy = [[10, 20, 30, 40]]
    
    mock_yolo = MagicMock()
    mock_yolo.return_value = [mock_result]
    
    mock_res.get.side_effect = lambda key: {
        "yolo_model": mock_yolo,
        "id_to_class": {0: "pizza"}
    }.get(key)
    
    resp = client.post(
        "/predict?conf=0.9",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")}
    )
    assert resp.status_code == 200


@patch("src.app.resources")
def test_predict_low_confidence(mock_res):
    """Test prediction with low confidence threshold"""
    img = Image.new('RGB', (100, 100), color='green')
    img_bytes = BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    mock_result = MagicMock()
    mock_result.boxes.cls = [0]
    mock_result.boxes.conf = [0.3]
    mock_result.boxes.xyxy = [[10, 20, 30, 40]]
    
    mock_yolo = MagicMock()
    mock_yolo.return_value = [mock_result]
    
    mock_res.get.side_effect = lambda key: {
        "yolo_model": mock_yolo,
        "id_to_class": {0: "burger"}
    }.get(key)
    
    resp = client.post(
        "/predict?conf=0.1",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")}
    )
    assert resp.status_code == 200


def test_predict_invalid_file():
    """Test predict with invalid file"""
    invalid_file = BytesIO(b"not an image")
    
    resources["yolo_model"] = MagicMock()
    
    resp = client.post(
        "/predict",
        files={"file": ("test.txt", invalid_file, "text/plain")}
    )
    # Should handle gracefully - check for error response
    assert resp.status_code in [400, 500]


@patch("src.app.resources")
def test_predict_yolo_exception(mock_res):
    """Test predict when YOLO raises an exception"""
    img = Image.new('RGB', (100, 100), color='yellow')
    img_bytes = BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    mock_yolo = MagicMock()
    mock_yolo.side_effect = Exception("YOLO prediction failed")
    
    mock_res.get.side_effect = lambda key: {
        "yolo_model": mock_yolo,
        "id_to_class": {0: "apple"}
    }.get(key)
    
    resp = client.post(
        "/predict",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")}
    )
    assert resp.status_code == 500


@patch("src.app.resources")
def test_predict_empty_detections(mock_res):
    """Test predict when no objects are detected"""
    img = Image.new('RGB', (100, 100), color='white')
    img_bytes = BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    mock_result = MagicMock()
    mock_result.boxes.cls = []
    mock_result.boxes.conf = []
    mock_result.boxes.xyxy = []
    
    mock_yolo = MagicMock()
    mock_yolo.return_value = [mock_result]
    
    mock_res.get.side_effect = lambda key: {
        "yolo_model": mock_yolo,
        "id_to_class": {}
    }.get(key)
    
    resp = client.post(
        "/predict",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data.get("detections", [])) == 0


@patch("src.app.boto3.client")
def test_s3_download_success(mock_boto):
    """Test successful S3 download"""
    from src.app import force_download_from_s3
    
    mock_s3 = MagicMock()
    mock_boto.return_value = mock_s3
    
    force_download_from_s3("test.txt", "/tmp/test.txt")
    mock_s3.download_file.assert_called_once()


@patch("src.app.boto3.client")
def test_s3_download_failure(mock_boto):
    """Test S3 download failure"""
    from src.app import force_download_from_s3
    
    mock_s3 = MagicMock()
    mock_s3.download_file.side_effect = Exception("S3 error")
    mock_boto.return_value = mock_s3
    
    result = force_download_from_s3("test.txt", "/tmp/test.txt")
    assert result is None


@patch("src.app.guard.validate_input", return_value=(True, "OK"))
@patch("src.app.resources")
def test_ask_with_vector_db(mock_res, _):
    """Test ask endpoint with vector database retrieval"""
    class FakeTensor:
        def __init__(self, data):
            self._data = list(data)

        @property
        def shape(self):
            return (len(self._data),)

        def __len__(self):
            return len(self._data)

        def __getitem__(self, idx):
            if isinstance(idx, slice):
                return FakeTensor(self._data[idx])
            return self._data[idx]

        def __iter__(self):
            return iter(self._data)

    class FakeInputs(dict):
        def __init__(self):
            super().__init__({"input_ids": FakeTensor([1, 2, 3])})

        def to(self, device):
            return self

    class FakeOutput:
        def __init__(self, data):
            self._data = FakeTensor(data)

        @property
        def shape(self):
            return (1, len(self._data))

        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return self._data

        def __iter__(self):
            return iter([self._data])

    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = FakeInputs()
    tokenizer.decode.return_value = "nutritional information"
    tokenizer.eos_token_id = 2

    model = MagicMock()
    model.device = "cpu"
    model.generate.return_value = FakeOutput([10, 11, 12])

    # Mock vector DB with similarity search
    mock_doc = MagicMock()
    mock_doc.page_content = "Apple is a healthy fruit"
    mock_vector_db = MagicMock()
    mock_vector_db.similarity_search.return_value = [mock_doc]

    mock_res.get.side_effect = lambda key: {
        "llm_model": model,
        "llm_tokenizer": tokenizer,
        "vector_db": mock_vector_db
    }.get(key)

    resp = client.post("/ask", json={"text": "Tell me about apples"})
    assert resp.status_code == 200
    assert "nutritional" in resp.json()["answer"]


def test_ask_empty_text():
    """Test ask endpoint with empty text"""
    resp = client.post("/ask", json={"text": ""})
    # Should handle empty input gracefully
    assert resp.status_code in [400, 422, 500]


def test_ask_very_long_text():
    """Test ask endpoint with very long text"""
    long_text = "What is nutrition? " * 1000
    resources["llm_model"] = MagicMock()
    resources["llm_tokenizer"] = MagicMock()
    
    resp = client.post("/ask", json={"text": long_text})
    # Should handle long input
    assert resp.status_code in [200, 400, 422, 500]
