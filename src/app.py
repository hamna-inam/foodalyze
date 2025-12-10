#!/usr/bin/env python3
"""
Foodalyze - Milestone 2 with HuggingFace LLMs
YOLO Food Detection + RAG Nutrition Q&A using lightweight LLM
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import uvicorn
import cv2
import numpy as np
import json
import logging
import time
from datetime import datetime
import os
from ultralytics import YOLO
import boto3  # <<< ADDED FOR S3

# --- RAG & LLM IMPORTS ---
import torch
import gc
from contextlib import asynccontextmanager
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM


logger = logging.getLogger(__name__)


# --- MONITORING ---
try:
    from starlette_prometheus import PrometheusMiddleware, metrics
    from prometheus_client import Counter, Histogram, REGISTRY

    MONITORING_ENABLED = True
    print("Monitoring enabled.")
except ImportError:
    MONITORING_ENABLED = False

# Global S3 client
s3 = boto3.client("s3")
S3_BUCKET = os.getenv(
    "S3_BUCKET_NAME", "foodalyze-artifacts-v2"
)  # <<< CHANGE THIS OR SET IN EC2 ENV


# ============================================================
# S3 DOWNLOAD HELPER
# ============================================================


def force_download_from_s3(local_path, s3_key):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    print(f"⬇️ Downloading from S3 → {local_path}")

    s3.download_file(S3_BUCKET, s3_key, local_path)
    print(f"   ✔ Downloaded {s3_key}")


# ============================================================================
# PROMETHEUS METRICS
# ============================================================================
if MONITORING_ENABLED:
    try:
        # Unregister any previously registered metrics
        metric_names = [
            "llm_token_usage_total",
            "guardrail_violations_total",
            "llm_request_duration_seconds",
            "yolo_inference_duration_seconds",
            "llm_cost_usd_total",
        ]
        for name in metric_names:
            try:
                if name in REGISTRY._names_to_collectors:
                    REGISTRY.unregister(REGISTRY._names_to_collectors[name])
            except Exception:

                pass

        LLM_TOKENS = Counter(
            "llm_token_usage_total",
            "Total LLM tokens used",
            ["type"],  # type = input or output
        )

        GUARDRAIL_VIOLATIONS = Counter(
            "guardrail_violations_total", "Total guardrail violations"
        )

        LLM_LATENCY = Histogram(
            "llm_request_duration_seconds",
            "LLM request latency in seconds",
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        )

        YOLO_LATENCY = Histogram(
            "yolo_inference_duration_seconds",
            "YOLO inference latency in seconds",
            buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
        )

        LLM_COST = Counter(
            "llm_cost_usd_total",
            "Total LLM cost in USD",
            ["token_type"],  # token_type = input or output
        )

    except Exception as e:
        logger.warning(f"Could not initialize metrics: {e}")
        MONITORING_ENABLED = False


# --- GUARDRAILS ---
try:
    from src.guardrails import guard
except ImportError:

    class DummyGuard:
        def validate_input(self, text):
            return True, "Safe"

        def validate_output(self, text):
            return True, "Safe"

    guard = DummyGuard()

# ============================================================================
# LOGGING
# ============================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
S3_YOLO_MODEL_KEY = "best.pt"
S3_CLASS_MAPPING_KEY = "class_mapping.json"
S3_FAISS_FAISS_KEY = "index.faiss"
S3_FAISS_PKL_KEY = "index.pkl"

LOCAL_YOLO_MODEL = "./tmp/models/best.pt"
LOCAL_CLASS_MAPPING = "./tmp/class_mapping.json"
LOCAL_FAISS_FAISS = "./tmp/faiss/index.faiss"
LOCAL_FAISS_PKL = "./tmp/faiss/index.pkl"

# --- LLM CONFIGURATION ---
LLM_MODEL_ID = "Qwen/Qwen2-0.5B-Instruct"
USE_4BIT = False

# ============================================================================
# PROMETHEUS METRICS
# ============================================================================
metric_names = [
    "llm_token_usage_total",
    "guardrail_violations_total",
    "llm_request_duration_seconds",
    "yolo_inference_duration_seconds",
]
for name in metric_names:
    if name in REGISTRY._names_to_collectors:
        try:
            REGISTRY.unregister(REGISTRY._names_to_collectors[name])
        except Exception:

            pass

LLM_TOKENS = Counter("llm_token_usage_total", "Total LLM tokens used", ["type"])
GUARDRAIL_VIOLATIONS = Counter("guardrail_violations_total", "Guardrail violations")
LLM_LATENCY = Histogram("llm_request_duration_seconds", "LLM request latency")
YOLO_LATENCY = Histogram("yolo_inference_duration_seconds", "YOLO inference latency")

# ============================================================================
# FOOD DATA (Trimmed for brevity, keep your full list)
# ============================================================================
FOOD_DATA = {
    "adhirasam": {
        "standard_portion_desc": "1 piece",
        "standard_portion_g": 35,
        "calories_per_100g": 337,
    },
    "aloo_gobi": {
        "standard_portion_desc": "1 bowl",
        "standard_portion_g": 150,
        "calories_per_100g": 80,
    },
    "aloo_matar": {
        "standard_portion_desc": "1 bowl",
        "standard_portion_g": 150,
        "calories_per_100g": 135,
    },
    "aloo_methi": {
        "standard_portion_desc": "1 bowl",
        "standard_portion_g": 150,
        "calories_per_100g": 151,
    },
    "aloo_shimla_mirch": {
        "standard_portion_desc": "1 bowl",
        "standard_portion_g": 150,
        "calories_per_100g": 138,
    },
    "aloo_tikki": {
        "standard_portion_desc": "1 piece",
        "standard_portion_g": 55,
        "calories_per_100g": 200,
    },
    "anarsa": {
        "standard_portion_desc": "1 piece",
        "standard_portion_g": 25,
        "calories_per_100g": 360,
    },
    "ariselu": {
        "standard_portion_desc": "1 piece",
        "standard_portion_g": 44,
        "calories_per_100g": 300,
    },
    "bandar_laddu": {
        "standard_portion_desc": "1 piece",
        "standard_portion_g": 35,
        "calories_per_100g": 356,
    },
    "basundi": {
        "standard_portion_desc": "1 bowl",
        "standard_portion_g": 150,
        "calories_per_100g": 152,
    },
    "bhatura": {
        "standard_portion_desc": "1 piece",
        "standard_portion_g": 50,
        "calories_per_100g": 400,
    },
    "bhindi_masala": {
        "standard_portion_desc": "1 bowl",
        "standard_portion_g": 150,
        "calories_per_100g": 107,
    },
    "biryani": {
        "standard_portion_desc": "1 plate",
        "standard_portion_g": 300,
        "calories_per_100g": 131,
    },
    "boondi": {
        "standard_portion_desc": "1 small bowl (savory)",
        "standard_portion_g": 30,
        "calories_per_100g": 584,
    },
    "butter_chicken": {
        "standard_portion_desc": "1 bowl",
        "standard_portion_g": 240,
        "calories_per_100g": 129,
    },
    "chak_hao_kheer": {
        "standard_portion_desc": "1 bowl",
        "standard_portion_g": 200,
        "calories_per_100g": 148,
    },
    "cham_cham": {
        "standard_portion_desc": "1 piece",
        "standard_portion_g": 50,
        "calories_per_100g": 240,
    },
    "chana_masala": {
        "standard_portion_desc": "1 bowl",
        "standard_portion_g": 240,
        "calories_per_100g": 145,
    },
    "chapati": {
        "standard_portion_desc": "1 chapati",
        "standard_portion_g": 40,
        "calories_per_100g": 300,
    },
    "chhena_kheeri": {
        "standard_portion_desc": "1 bowl",
        "standard_portion_g": 150,
        "calories_per_100g": 231,
    },
    "chicken_razala": {
        "standard_portion_desc": "1 bowl",
        "standard_portion_g": 235,
        "calories_per_100g": 140,
    },
    "chicken_tikka": {
        "standard_portion_desc": "1 serving (6 pieces)",
        "standard_portion_g": 150,
        "calories_per_100g": 179,
    },
    "chicken_tikka_masala": {
        "standard_portion_desc": "1 bowl",
        "standard_portion_g": 240,
        "calories_per_100g": 191,
    },
    "chikki": {
        "standard_portion_desc": "1 small bar",
        "standard_portion_g": 20,
        "calories_per_100g": 520,
    },
    "daal_baati_churma": {
        "standard_portion_desc": "1 serving (1 baati, 1 cup dal)",
        "standard_portion_g": 250,
        "calories_per_100g": 203,
    },
    "daal_puri": {
        "standard_portion_desc": "1 puri",
        "standard_portion_g": 45,
        "calories_per_100g": 270,
    },
    "dal_makhani": {
        "standard_portion_desc": "1 bowl",
        "standard_portion_g": 250,
        "calories_per_100g": 111,
    },
    "dal_tadka": {
        "standard_portion_desc": "1 bowl",
        "standard_portion_g": 250,
        "calories_per_100g": 179,
    },
    "dharwad_pedha": {
        "standard_portion_desc": "1 piece",
        "standard_portion_g": 26,
        "calories_per_100g": 446,
    },
    "doodhpak": {
        "standard_portion_desc": "1 bowl",
        "standard_portion_g": 200,
        "calories_per_100g": 212,
    },
    "double_ka_meetha": {
        "standard_portion_desc": "1 piece",
        "standard_portion_g": 75,
        "calories_per_100g": 385,
    },
    "dum_aloo": {
        "standard_portion_desc": "1 bowl",
        "standard_portion_g": 240,
        "calories_per_100g": 170,
    },
    "gajar_ka_halwa": {
        "standard_portion_desc": "1 bowl",
        "standard_portion_g": 100,
        "calories_per_100g": 345,
    },
    "gavvalu": {
        "standard_portion_desc": "1 serving",
        "standard_portion_g": 30,
        "calories_per_100g": 422,
    },
    "ghevar": {
        "standard_portion_desc": "1 piece",
        "standard_portion_g": 150,
        "calories_per_100g": 445,
    },
    "gulab_jamun": {
        "standard_portion_desc": "1 piece",
        "standard_portion_g": 45,
        "calories_per_100g": 286,
    },
    "imarti": {
        "standard_portion_desc": "1 piece",
        "standard_portion_g": 25,
        "calories_per_100g": 356,
    },
    "jalebi": {
        "standard_portion_desc": "1 piece",
        "standard_portion_g": 40,
        "calories_per_100g": 450,
    },
    "kachori": {
        "standard_portion_desc": "1 piece",
        "standard_portion_g": 55,
        "calories_per_100g": 513,
    },
    "kadai_paneer": {
        "standard_portion_desc": "1 bowl",
        "standard_portion_g": 240,
        "calories_per_100g": 130,
    },
    "kadhi_pakoda": {
        "standard_portion_desc": "1 bowl",
        "standard_portion_g": 250,
        "calories_per_100g": 79,
    },
    "kajjikaya": {
        "standard_portion_desc": "1 piece (karanji)",
        "standard_portion_g": 70,
        "calories_per_100g": 350,
    },
    "kakinada_khaja": {
        "standard_portion_desc": "1 piece",
        "standard_portion_g": 34,
        "calories_per_100g": 297,
    },
    "kalakand": {
        "standard_portion_desc": "1 piece",
        "standard_portion_g": 44,
        "calories_per_100g": 386,
    },
    "karela_bharta": {
        "standard_portion_desc": "1 serving",
        "standard_portion_g": 150,
        "calories_per_100g": 218,
    },
    "kofta": {
        "standard_portion_desc": "1 bowl (malai kofta)",
        "standard_portion_g": 250,
        "calories_per_100g": 148,
    },
    "kuzhi_paniyaram": {
        "standard_portion_desc": "1 piece",
        "standard_portion_g": 30,
        "calories_per_100g": 207,
    },
    "lassi": {
        "standard_portion_desc": "1 glass",
        "standard_portion_g": 180,
        "calories_per_100g": 113,
    },
    "ledikeni": {
        "standard_portion_desc": "1 piece",
        "standard_portion_g": 40,
        "calories_per_100g": 350,
    },
    "litti_chokha": {
        "standard_portion_desc": "1 serving (2 litti, 1 bowl chokha)",
        "standard_portion_g": 240,
        "calories_per_100g": 138,
    },
    "lyangcha": {
        "standard_portion_desc": "1 piece",
        "standard_portion_g": 40,
        "calories_per_100g": 350,
    },
    "maach_jhol": {
        "standard_portion_desc": "1 bowl",
        "standard_portion_g": 250,
        "calories_per_100g": 85,
    },
    "makki_di_roti_sarson_da_saag": {
        "standard_portion_desc": "1 roti, 1 bowl saag",
        "standard_portion_g": 200,
        "calories_per_100g": 150,
    },
    "malapua": {
        "standard_portion_desc": "1 piece",
        "standard_portion_g": 50,
        "calories_per_100g": 350,
    },
    "misi_roti": {
        "standard_portion_desc": "1 roti",
        "standard_portion_g": 60,
        "calories_per_100g": 306,
    },
    "misti_doi": {
        "standard_portion_desc": "1 small cup",
        "standard_portion_g": 100,
        "calories_per_100g": 190,
    },
    "modak": {
        "standard_portion_desc": "1 piece",
        "standard_portion_g": 40,
        "calories_per_100g": 288,
    },
    "mysore_pak": {
        "standard_portion_desc": "1 piece",
        "standard_portion_g": 40,
        "calories_per_100g": 618,
    },
    "naan": {
        "standard_portion_desc": "1 naan",
        "standard_portion_g": 100,
        "calories_per_100g": 336,
    },
    "navrattan_korma": {
        "standard_portion_desc": "1 bowl",
        "standard_portion_g": 240,
        "calories_per_100g": 154,
    },
    "palak_paneer": {
        "standard_portion_desc": "1 bowl",
        "standard_portion_g": 240,
        "calories_per_100g": 198,
    },
    "paneer_butter_masala": {
        "standard_portion_desc": "1 bowl",
        "standard_portion_g": 240,
        "calories_per_100g": 150,
    },
    "phirni": {
        "standard_portion_desc": "1 small cup",
        "standard_portion_g": 100,
        "calories_per_100g": 243,
    },
    "pithe": {
        "standard_portion_desc": "1 piece (patishapta)",
        "standard_portion_g": 60,
        "calories_per_100g": 233,
    },
    "poha": {
        "standard_portion_desc": "1 bowl",
        "standard_portion_g": 150,
        "calories_per_100g": 180,
    },
    "poornalu": {
        "standard_portion_desc": "1 piece",
        "standard_portion_g": 30,
        "calories_per_100g": 350,
    },
    "pootharekulu": {
        "standard_portion_desc": "1 piece",
        "standard_portion_g": 40,
        "calories_per_100g": 438,
    },
    "qubani_ka_meetha": {
        "standard_portion_desc": "1 small cup",
        "standard_portion_g": 100,
        "calories_per_100g": 324,
    },
    "rabri": {
        "standard_portion_desc": "1 serving",
        "standard_portion_g": 100,
        "calories_per_100g": 275,
    },
    "ras_malai": {
        "standard_portion_desc": "1 small cup (2 pieces)",
        "standard_portion_g": 100,
        "calories_per_100g": 163,
    },
    "rasgulla": {
        "standard_portion_desc": "1 piece",
        "standard_portion_g": 50,
        "calories_per_100g": 186,
    },
    "sandesh": {
        "standard_portion_desc": "1 piece",
        "standard_portion_g": 50,
        "calories_per_100g": 250,
    },
    "shankarpali": {
        "standard_portion_desc": "1 serving",
        "standard_portion_g": 30,
        "calories_per_100g": 422,
    },
    "sheer_korma": {
        "standard_portion_desc": "1 small cup",
        "standard_portion_g": 100,
        "calories_per_100g": 288,
    },
    "sheera": {
        "standard_portion_desc": "1 bowl (suji halwa)",
        "standard_portion_g": 100,
        "calories_per_100g": 360,
    },
    "shrikhand": {
        "standard_portion_desc": "1 small cup",
        "standard_portion_g": 100,
        "calories_per_100g": 234,
    },
    "sohan_halwa": {
        "standard_portion_desc": "1 piece",
        "standard_portion_g": 30,
        "calories_per_100g": 445,
    },
    "sohan_papdi": {
        "standard_portion_desc": "1 piece",
        "standard_portion_g": 21,
        "calories_per_100g": 476,
    },
    "sutar_feni": {
        "standard_portion_desc": "1 serving",
        "standard_portion_g": 50,
        "calories_per_100g": 476,
    },
    "unni_appam": {
        "standard_portion_desc": "1 piece",
        "standard_portion_g": 55,
        "calories_per_100g": 345,
    },
}

# ============================================================================
# GLOBAL RESOURCES
# ============================================================================
resources = {}


# ============================================================================
# LIFESPAN: Load models on startup (FIX APPLIED HERE)
# ============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting Foodalyze (YOLO + HuggingFace LLM)...")

    # S3 → Local syncing
    print("\n📦 Fetching artifacts from S3...\n")
    force_download_from_s3(LOCAL_YOLO_MODEL, S3_YOLO_MODEL_KEY)
    force_download_from_s3(LOCAL_CLASS_MAPPING, S3_CLASS_MAPPING_KEY)
    force_download_from_s3(LOCAL_FAISS_FAISS, S3_FAISS_FAISS_KEY)
    force_download_from_s3(LOCAL_FAISS_PKL, S3_FAISS_PKL_KEY)
    # --- Load YOLO Model ---
    try:
        resources["yolo_model"] = YOLO(LOCAL_YOLO_MODEL)
        logger.info(f"✅ YOLO model loaded: {LOCAL_YOLO_MODEL}")
    except Exception as e:
        logger.error(f"❌ Failed to load YOLO: {e}")
        resources["yolo_model"] = None

    # --- Load Class Mapping ---
    try:
        with open(LOCAL_CLASS_MAPPING, "r") as f:
            class_mapping = json.load(f)
            resources["class_to_id"] = class_mapping["class_to_id"]
            resources["id_to_class"] = {
                int(k): v for k, v in class_mapping["id_to_class"].items()
            }
        logger.info(
            f"✅ Class mapping loaded ({len(resources['id_to_class'])} classes)"
        )
    except Exception as e:
        logger.error(f"❌ Failed to load class mapping: {e}")
        resources["id_to_class"] = {}

    # --- Load Vector DB (FAISS) ---
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        resources["vector_db"] = FAISS.load_local(
            "./tmp/faiss", embeddings, allow_dangerous_deserialization=True
        )
        logger.info("✅ Vector DB loaded from S3 artifacts (./tmp/faiss)")
    except Exception as e:
        logger.warning(f"⚠️ Vector DB not available: {e}")
        resources["vector_db"] = None

    # --- Load LLM (HuggingFace) ---
    try:
        gc.collect()

        # Determine the correct device (FIX: Explicitly check and set device)
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        logger.info(f"Using device: {device}")

        # Explicitly clear cache if using a GPU device
        if device.type != "cpu":
            if device.type == "mps":
                torch.mps.empty_cache()
            else:
                torch.cuda.empty_cache()

        logger.info(f"Loading {LLM_MODEL_ID}...")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID, trust_remote_code=True)

        # Load model without quantization
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_ID,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            # CRITICAL: We remove device_map="auto" to explicitly move the model ourselves
        )

        # 🔥 CRITICAL FIX: Explicitly move the model to the determined device
        # This resolves the "device meta" error by ensuring all components are on mps:0
        model.to(device)

        resources["llm_tokenizer"] = tokenizer
        resources["llm_model"] = model
        resources["llm_device"] = device  # Store the device for later use
        logger.info(f"✅ LLM loaded successfully on {device}: {LLM_MODEL_ID}")
    except Exception as e:
        logger.error(f"❌ Failed to load LLM: {e}")
        resources["llm_model"] = None
        resources["llm_tokenizer"] = None
        resources["llm_device"] = torch.device("cpu")  # Default if loading fails

    yield

    # --- Cleanup on shutdown ---
    logger.info("🛑 Shutting down...")
    resources.clear()
    gc.collect()
    try:
        # Use device-aware cleanup
        if resources.get("llm_device", torch.device("cpu")).type == "mps":
            torch.mps.empty_cache()
        elif resources.get("llm_device", torch.device("cpu")).type == "cuda":
            torch.cuda.empty_cache()
    except Exception as e:
        logger.warning(f"Cleanup error: {e}")


# ============================================================================
# CREATE FASTAPI APP
# ============================================================================
app = FastAPI(
    title="Foodalyze - YOLO + HuggingFace LLM API",
    description="Detect Indian food dishes + answer nutrition questions",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(PrometheusMiddleware)
app.add_route("/metrics", metrics)


# ============================================================================
# PYDANTIC MODELS
# ============================================================================
class QueryRequest(BaseModel):
    text: str


class HealthResponse(BaseModel):
    status: str
    yolo_loaded: bool
    rag_loaded: bool
    llm_loaded: bool
    timestamp: str


# ============================================================================
# ENDPOINTS
# ============================================================================
@app.get("/health", tags=["Health"])
def health_check() -> HealthResponse:
    return HealthResponse(
        status="OK",
        yolo_loaded=resources.get("yolo_model") is not None,
        rag_loaded=resources.get("vector_db") is not None,
        llm_loaded=resources.get("llm_model") is not None,
        timestamp=datetime.now().isoformat(),
    )


@app.get("/", tags=["Info"])
def root():
    return {
        "name": "Foodalyze API",
        "version": "2.0.0",
        "endpoints": {
            "docs": "/docs",
            "food_detection": "/",
            "nutrition_qa": "/ask",
            "health": "/health",
            "model_info": "/model_info",
            "metrics": "/metrics",
        },
    }


@app.get("/model_info", tags=["Info"])
def model_info():
    return {
        "yolo_model": LOCAL_YOLO_MODEL,
        "num_classes": len(resources.get("id_to_class", {})),
        "classes": resources.get("id_to_class", {}),
        "llm_model": LLM_MODEL_ID if resources.get("llm_model") else "Not loaded",
        "vector_db": "FAISS" if resources.get("vector_db") else "Not loaded",
    }


def safe_decode_image(contents: bytes):
    """Decode image or raise HTTP 400 cleanly."""
    try:
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    return image


@app.post("/predict", tags=["Food Detection"])
async def predict(file: UploadFile = File(...), conf: float = 0.5):
    yolo_model = resources.get("yolo_model")
    id_to_class = resources.get("id_to_class", {})

    if yolo_model is None:
        raise HTTPException(status_code=500, detail="YOLO model not loaded")

    try:
        # Read uploaded file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)

        # ---- INNER TRY BLOCK (catches decode errors correctly) ----
        try:
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                raise HTTPException(status_code=400, detail="Invalid image file")
        except Exception:
            # Any decode failure (including mock_imdecode) returns 400
            raise HTTPException(status_code=400, detail="Invalid image file")

        # ---- YOLO INFERENCE ----
        start = time.time()
        results = yolo_model.predict(source=image, conf=conf, verbose=False)
        duration = time.time() - start

        result = results[0]
        detections = []

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = id_to_class.get(class_id, f"unknown_{class_id}")

            food_info = FOOD_DATA.get(class_name)
            if food_info:
                portion = food_info["standard_portion_desc"]
                portion_g = food_info["standard_portion_g"]
                calories = round((portion_g * food_info["calories_per_100g"]) / 100, 2)
            else:
                portion = "Unknown"
                portion_g = None
                calories = None

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(confidence, 4),
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "portion_desc": portion,
                    "portion_g": portion_g,
                    "calories_estimate": calories,
                }
            )

        return {
            "image": file.filename,
            "num_detections": len(detections),
            "detections": detections,
            "inference_time_ms": round(duration * 1000, 2),
            "timestamp": datetime.now().isoformat(),
        }

    except HTTPException:
        # Allows 400 errors to pass through unchanged
        raise

    except Exception as e:
        logger.error(f"Prediction unexpected error: {e}")
        raise HTTPException(status_code=400, detail="Prediction failed")


@app.post("/ask", tags=["RAG Q&A"])
def ask(q: QueryRequest):
    """Ask a question about Indian food nutrition using HuggingFace LLM."""
    is_valid, msg = guard.validate_input(q.text)
    if not is_valid:
        GUARDRAIL_VIOLATIONS.inc()
        raise HTTPException(status_code=400, detail=msg)
    if not is_valid and MONITORING_ENABLED:
        try:
            print("Logging guardrail violation...")
            GUARDRAIL_VIOLATIONS.inc()
        except Exception:

            print("Failed to log guardrail violation.")
            pass

    llm_model = resources.get("llm_model")
    llm_tokenizer = resources.get("llm_tokenizer")
    vector_db = resources.get("vector_db")

    if llm_model is None or llm_tokenizer is None:
        raise HTTPException(status_code=500, detail="LLM not loaded")

    # --- Retrieve context from FAISS ---
    context = "No context available."
    if vector_db:
        try:
            docs = vector_db.similarity_search(q.text, k=3)
            context = "\n".join([d.page_content for d in docs])
        except Exception as e:
            logger.warning(f"Retrieval error: {e}")

    # --- Generate response ---
    start_time = time.time()
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a hyper-efficient, expert nutritionist specializing in Indian cuisine. "
                    "Your sole purpose is to provide direct, factual, and concise answers."
                    "\n\n### FORMATTING RULES ###"
                    "\n1. **OUTPUT MUST BE ONE SINGLE PARAGRAPH.**"
                    "\n2. DO NOT use introductory phrases (e.g., 'Sure, here is...', 'The following tips...'). START IMMEDIATELY with the answer content."
                    "\n3. NEVER use list formatting (e.g., numbered lists or bullet points)."
                    "\n4. DO NOT use concluding remarks or summaries."
                    "\n5. NEVER let your answer end abruptly; finish sentences completely."
                    "\n6. Never mention that you are an AI model or refer to yourself in any way."
                    "\n\nRespond ONLY to the question using the constraints above."
                ),
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {q.text}"},
        ]

        # Use the model's device, which was explicitly set in the lifespan function
        inputs = llm_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(
            llm_model.device
        )  # Ensures input tensors are on the correct device (mps:0)

        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=400,
            do_sample=True,
            temperature=0.5,
            top_p=0.9,
            pad_token_id=llm_tokenizer.eos_token_id,
            eos_token_id=llm_tokenizer.eos_token_id,
        )

        answer = llm_tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        ).strip()
        # --- Track tokens and cost ---
        if MONITORING_ENABLED:
            try:
                print("Tracking tokens and cost...")
                input_tokens = inputs["input_ids"].shape[-1]
                output_tokens = outputs.shape[-1] - input_tokens

                LLM_TOKENS.labels(type="input").inc(input_tokens)
                LLM_TOKENS.labels(type="output").inc(output_tokens)

                # Cost tracking (TinyLlama is free, but we track it anyway)
                COST_PER_INPUT_1K = 0.0
                COST_PER_OUTPUT_1K = 0.0
                input_cost = (input_tokens / 1000.0) * COST_PER_INPUT_1K
                output_cost = (output_tokens / 1000.0) * COST_PER_OUTPUT_1K

                if input_cost > 0:
                    LLM_COST.labels(token_type="input").inc(input_cost)
                if output_cost > 0:
                    LLM_COST.labels(token_type="output").inc(output_cost)
            except Exception as e:
                logger.warning(f"Error tracking tokens: {e}")

        # Track tokens
        input_tokens = inputs["input_ids"].shape[-1]
        output_tokens = outputs.shape[-1] - input_tokens
        LLM_TOKENS.labels(type="input").inc(input_tokens)
        LLM_TOKENS.labels(type="output").inc(output_tokens)

    except Exception as e:
        logger.error(f"Generation error: {e}")
        answer = f"Error: {str(e)}"

    duration = time.time() - start_time

    if MONITORING_ENABLED:
        try:
            print("Logging LLM latency...")
            LLM_LATENCY.observe(duration)
        except Exception:
            pass

    # --- Output validation (Guardrails) ---

    is_safe, msg = guard.validate_output(answer)
    if not is_safe:
        if MONITORING_ENABLED:
            try:
                print("Logging guardrail violation...")
                GUARDRAIL_VIOLATIONS.inc()
            except Exception:

                print("Failed to log guardrail violation.")
        answer = "[Safety Filter] Response blocked due to safety policy."

    return {
        "question": q.text,
        "answer": answer,
        "context": context,
        "inference_time_ms": round(duration * 1000, 2),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/debug/guardrail_stats", tags=["Debug"])
def guardrail_stats():
    """Get current guardrail metrics."""
    return {
        "guardrail_violations": GUARDRAIL_VIOLATIONS._value.get(),
    }


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
