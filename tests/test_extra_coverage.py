from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from src.app import app, resources

client = TestClient(app)


def test_yolo_load_failure():
    with patch("src.app.YOLO", side_effect=Exception("load error")):
        with patch("src.app.force_download_from_s3"):
            with patch("src.app.AutoTokenizer"):
                with patch("src.app.AutoModelForCausalLM"):
                    # trigger startup
                    with TestClient(app) as test_client:
                        assert resources.get("yolo_model") is None
def test_class_mapping_load_failure():
    with patch("builtins.open", side_effect=Exception("file error")):
        with TestClient(app) as c:
            assert resources.get("id_to_class") == {}
def test_faiss_load_failure():
    with patch("src.app.FAISS.load_local", side_effect=Exception("FAISS error")):
        with TestClient(app) as c:
            assert resources.get("vector_db") is None
def test_llm_load_failure():
    with patch("src.app.AutoTokenizer.from_pretrained", side_effect=Exception("LLM error")):
        with TestClient(app) as c:
            assert resources.get("llm_model") is None
def test_ask_llm_not_loaded():
    resources["llm_model"] = None
    resources["llm_tokenizer"] = None

    resp = client.post("/ask", json={"text": "Hello"})
    assert resp.status_code == 500
  
@patch("src.app.guard.validate_input", return_value=(False, "Blocked"))
def test_ask_guardrail_block(_,):
    resp = client.post("/ask", json={"text": "bad request"})
    assert resp.status_code == 400

@patch("src.app.guard.validate_input", return_value=(True, "OK"))
@patch("src.app.resources")
def test_ask_generation_failure(mock_res):
    mock_res.get.return_value = MagicMock()
    mock_res.get("vector_db").similarity_search.side_effect = Exception("fail")
    mock_res.get("llm_model").generate.side_effect = Exception("gen error")

    resp = client.post("/ask", json={"text": "nutrition info"})
    assert resp.status_code == 200
    assert "Error:" in resp.json()["answer"]
@patch("src.app.guard.validate_input", return_value=(True, "OK"))
def test_ask_success(_):
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = MagicMock(to=lambda x: x)
    tokenizer.decode.return_value = "healthy choice"

    model = MagicMock()
    model.device = "cpu"
    model.generate.return_value = [[1, 2, 3, 4]]

    resources["llm_model"] = model
    resources["llm_tokenizer"] = tokenizer
    resources["vector_db"] = None

    resp = client.post("/ask", json={"text": "is biryani healthy?"})
    assert resp.status_code == 200
    assert "healthy" in resp.json()["answer"]

