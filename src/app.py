import os
import torch
import gc
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, REGISTRY

# Import Guardrails
from src.guardrails import guard 

# Metrics Cleanup (Prevents "Duplicated Timeseries" Error)
metric_names = ["llm_token_usage_total", "guardrail_violations_total", "llm_request_duration_seconds"]
for name in metric_names:
    if name in REGISTRY._names_to_collectors:
        REGISTRY.unregister(REGISTRY._names_to_collectors[name])

# Metrics
TOKEN_USAGE = Counter("llm_token_usage_total", "Total tokens used", ["type"])
GUARDRAIL_VIOLATIONS = Counter("guardrail_violations_total", "Blocked requests")
REQUEST_LATENCY = Histogram("llm_request_duration_seconds", "LLM inference time")

resources = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Initializing AI...")
    # Load DB
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        resources["vector_db"] = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except: resources["vector_db"] = None

    # Load DeepSeek
    try:
        gc.collect(); torch.cuda.empty_cache()
        model_id = "deepseek-ai/deepseek-llm-7b-chat"
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0}, trust_remote_code=True)
        resources["tokenizer"] = tokenizer
        resources["model"] = model
    except: resources["model"] = None
    yield
    resources.clear()

app = FastAPI(lifespan=lifespan)
Instrumentator().instrument(app).expose(app)

class Query(BaseModel):
    text: str

@app.post("/ask")
def ask(q: Query):
    # --- CHECK INPUT ---
    is_valid, msg = guard.validate_input(q.text)
    if not is_valid:
        GUARDRAIL_VIOLATIONS.inc()
        # Return 400 Bad Request to indicate BLOCK
        raise HTTPException(status_code=400, detail=msg)

    # --- NORMAL RAG ---
    vector_db = resources.get("vector_db")
    model = resources.get("model")
    tokenizer = resources.get("tokenizer")

    # Mock RAG response for speed testing (DeepSeek takes time)
    answer = "This is a response."
    context = "Context"

    if model and tokenizer:
         # Real generation logic here
         pass 

    return {"answer": answer, "context": context}
