#!/usr/bin/env python3
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime
from ultralytics import YOLO

# --- ADD THESE 3 LINES FOR MONITORING (D5) ---
from starlette_prometheus import PrometheusMiddleware, metrics
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Food Detection API",
    description="YOLO-based API for detecting Indian food dishes with calorie info",
    version="1.1.0"
)

# --- ADD THESE 2 LINES FOR MONITORING (D5) ---
app.add_middleware(PrometheusMiddleware)
app.add_route("/metrics", metrics)

# --- PATHS CORRECTED FOR OUR PROJECT STRUCTURE ---
MODEL_PATH = "./models/best.pt"
CLASS_MAPPING_PATH = "./class_mapping.json"

# ---- Calorie lookup table ----
FOOD_DATA = {
    'adhirasam': {'standard_portion_desc': '1 piece', 'standard_portion_g': 35, 'calories_per_100g': 337},
    'aloo_gobi': {'standard_portion_desc': '1 bowl', 'standard_portion_g': 150, 'calories_per_100g': 80},
    'aloo_matar': {'standard_portion_desc': '1 bowl', 'standard_portion_g': 150, 'calories_per_100g': 135},
    'aloo_methi': {'standard_portion_desc': '1 bowl', 'standard_portion_g': 150, 'calories_per_100g': 151},
    'aloo_shimla_mirch': {'standard_portion_desc': '1 bowl', 'standard_portion_g': 150, 'calories_per_100g': 138},
    'aloo_tikki': {'standard_portion_desc': '1 piece', 'standard_portion_g': 55, 'calories_per_100g': 200},
    'anarsa': {'standard_portion_desc': '1 piece', 'standard_portion_g': 25, 'calories_per_100g': 360},
    'ariselu': {'standard_portion_desc': '1 piece', 'standard_portion_g': 44, 'calories_per_100g': 300},
    'bandar_laddu': {'standard_portion_desc': '1 piece', 'standard_portion_g': 35, 'calories_per_100g': 356},
    'basundi': {'standard_portion_desc': '1 bowl', 'standard_portion_g': 150, 'calories_per_100g': 152},
    'bhatura': {'standard_portion_desc': '1 piece', 'standard_portion_g': 50, 'calories_per_100g': 400},
    'bhindi_masala': {'standard_portion_desc': '1 bowl', 'standard_portion_g': 150, 'calories_per_100g': 107},
    'biryani': {'standard_portion_desc': '1 plate', 'standard_portion_g': 300, 'calories_per_100g': 131},
    'boondi': {'standard_portion_desc': '1 small bowl (savory)', 'standard_portion_g': 30, 'calories_per_100g': 584},
    'butter_chicken': {'standard_portion_desc': '1 bowl', 'standard_portion_g': 240, 'calories_per_100g': 129},
    'chak_hao_kheer': {'standard_portion_desc': '1 bowl', 'standard_portion_g': 200, 'calories_per_100g': 148},
    'cham_cham': {'standard_portion_desc': '1 piece', 'standard_portion_g': 50, 'calories_per_100g': 240},
    'chana_masala': {'standard_portion_desc': '1 bowl', 'standard_portion_g': 240, 'calories_per_100g': 145},
    'chapati': {'standard_portion_desc': '1 chapati', 'standard_portion_g': 40, 'calories_per_100g': 300},
    'chhena_kheeri': {'standard_portion_desc': '1 bowl', 'standard_portion_g': 150, 'calories_per_100g': 231},
    'chicken_razala': {'standard_portion_desc': '1 bowl', 'standard_portion_g': 235, 'calories_per_100g': 140},
    'chicken_tikka': {'standard_portion_desc': '1 serving (6 pieces)', 'standard_portion_g': 150, 'calories_per_100g': 179},
    'chicken_tikka_masala': {'standard_portion_desc': '1 bowl', 'standard_portion_g': 240, 'calories_per_100g': 191},
    'chikki': {'standard_portion_desc': '1 small bar', 'standard_portion_g': 20, 'calories_per_100g': 520},
    'daal_baati_churma': {'standard_portion_desc': '1 serving (1 baati, 1 cup dal)', 'standard_portion_g': 250, 'calories_per_100g': 203},
    'daal_puri': {'standard_portion_desc': '1 puri', 'standard_portion_g': 45, 'calories_per_100g': 270},
    'dal_makhani': {'standard_portion_desc': '1 bowl', 'standard_portion_g': 250, 'calories_per_100g': 111},
    'dal_tadka': {'standard_portion_desc': '1 bowl', 'standard_portion_g': 250, 'calories_per_100g': 179},
    'dharwad_pedha': {'standard_portion_desc': '1 piece', 'standard_portion_g': 26, 'calories_per_100g': 446},
    'doodhpak': {'standard_portion_desc': '1 bowl', 'standard_portion_g': 200, 'calories_per_100g': 212},
    'double_ka_meetha': {'standard_portion_desc': '1 piece', 'standard_portion_g': 75, 'calories_per_100g': 385},
    'dum_aloo': {'standard_portion_desc': '1 bowl', 'standard_portion_g': 240, 'calories_per_100g': 170},
    'gajar_ka_halwa': {'standard_portion_desc': '1 bowl', 'standard_portion_g': 100, 'calories_per_100g': 345},
    'gavvalu': {'standard_portion_desc': '1 serving', 'standard_portion_g': 30, 'calories_per_100g': 422},
    'ghevar': {'standard_portion_desc': '1 piece', 'standard_portion_g': 150, 'calories_per_100g': 445},
    'gulab_jamun': {'standard_portion_desc': '1 piece', 'standard_portion_g': 45, 'calories_per_100g': 286},
    'imarti': {'standard_portion_desc': '1 piece', 'standard_portion_g': 25, 'calories_per_100g': 356},
    'jalebi': {'standard_portion_desc': '1 piece', 'standard_portion_g': 40, 'calories_per_100g': 450},
    'kachori': {'standard_portion_desc': '1 piece', 'standard_portion_g': 55, 'calories_per_100g': 513},
    'kadai_paneer': {'standard_portion_desc': '1 bowl', 'standard_portion_g': 240, 'calories_per_100g': 130},
    'kadhi_pakoda': {'standard_portion_desc': '1 bowl', 'standard_portion_g': 250, 'calories_per_100g': 79},
    'kajjikaya': {'standard_portion_desc': '1 piece (karanji)', 'standard_portion_g': 70, 'calories_per_100g': 350},
    'kakinada_khaja': {'standard_portion_desc': '1 piece', 'standard_portion_g': 34, 'calories_per_100g': 297},
    'kalakand': {'standard_portion_desc': '1 piece', 'standard_portion_g': 44, 'calories_per_100g': 386},
    'karela_bharta': {'standard_portion_desc': '1 serving', 'standard_portion_g': 150, 'calories_per_100g': 218},
    'kofta': {'standard_portion_desc': '1 bowl (malai kofta)', 'standard_portion_g': 250, 'calories_per_100g': 148},
    'kuzhi_paniyaram': {'standard_portion_desc': '1 piece', 'standard_portion_g': 30, 'calories_per_100g': 207},
    'lassi': {'standard_portion_desc': '1 glass', 'standard_portion_g': 180, 'calories_per_100g': 113},
    'ledikeni': {'standard_portion_desc': '1 piece', 'standard_portion_g': 40, 'calories_per_100g': 350},
    'litti_chokha': {'standard_portion_desc': '1 serving (2 litti, 1 bowl chokha)', 'standard_portion_g': 240, 'calories_per_100g': 138},
    'lyangcha': {'standard_portion_desc': '1 piece', 'standard_portion_g': 40, 'calories_per_100g': 350},
    'maach_jhol': {'standard_portion_desc': '1 bowl', 'standard_portion_g': 250, 'calories_per_100g': 85},
    'makki_di_roti_sarson_da_saag': {'standard_portion_desc': '1 roti, 1 bowl saag', 'standard_portion_g': 200, 'calories_per_100g': 150},
    'malapua': {'standard_portion_desc': '1 piece', 'standard_portion_g': 50, 'calories_per_100g': 350},
    'misi_roti': {'standard_portion_desc': '1 roti', 'standard_portion_g': 60, 'calories_per_100g': 306},
    'misti_doi': {'standard_portion_desc': '1 small cup', 'standard_portion_g': 100, 'calories_per_100g': 190},
    'modak': {'standard_portion_desc': '1 piece', 'standard_portion_g': 40, 'calories_per_100g': 288},
    'mysore_pak': {'standard_portion_desc': '1 piece', 'standard_portion_g': 40, 'calories_per_100g': 618},
    'naan': {'standard_portion_desc': '1 naan', 'standard_portion_g': 100, 'calories_per_100g': 336},
    'navrattan_korma': {'standard_portion_desc': '1 bowl', 'standard_portion_g': 240, 'calories_per_100g': 154},
    'palak_paneer': {'standard_portion_desc': '1 bowl', 'standard_portion_g': 240, 'calories_per_100g': 198},
    'paneer_butter_masala': {'standard_portion_desc': '1 bowl', 'standard_portion_g': 240, 'calories_per_100g': 150},
    'phirni': {'standard_portion_desc': '1 small cup', 'standard_portion_g': 100, 'calories_per_100g': 243},
    'pithe': {'standard_portion_desc': '1 piece (patishapta)', 'standard_portion_g': 60, 'calories_per_100g': 233},
    'poha': {'standard_portion_desc': '1 bowl', 'standard_portion_g': 150, 'calories_per_100g': 180},
    'poornalu': {'standard_portion_desc': '1 piece', 'standard_portion_g': 30, 'calories_per_100g': 350},
    'pootharekulu': {'standard_portion_desc': '1 piece', 'standard_portion_g': 40, 'calories_per_100g': 438},
    'qubani_ka_meetha': {'standard_portion_desc': '1 small cup', 'standard_portion_g': 100, 'calories_per_100g': 324},
    'rabri': {'standard_portion_desc': '1 serving', 'standard_portion_g': 100, 'calories_per_100g': 275},
    'ras_malai': {'standard_portion_desc': '1 small cup (2 pieces)', 'standard_portion_g': 100, 'calories_per_100g': 163},
    'rasgulla': {'standard_portion_desc': '1 piece', 'standard_portion_g': 50, 'calories_per_100g': 186},
    'sandesh': {'standard_portion_desc': '1 piece', 'standard_portion_g': 50, 'calories_per_100g': 250},
    'shankarpali': {'standard_portion_desc': '1 serving', 'standard_portion_g': 30, 'calories_per_100g': 422},
    'sheer_korma': {'standard_portion_desc': '1 small cup', 'standard_portion_g': 100, 'calories_per_100g': 288},
    'sheera': {'standard_portion_desc': '1 bowl (suji halwa)', 'standard_portion_g': 100, 'calories_per_100g': 360},
    'shrikhand': {'standard_portion_desc': '1 small cup', 'standard_portion_g': 100, 'calories_per_100g': 234},
    'sohan_halwa': {'standard_portion_desc': '1 piece', 'standard_portion_g': 30, 'calories_per_100g': 445},
    'sohan_papdi': {'standard_portion_desc': '1 piece', 'standard_portion_g': 21, 'calories_per_100g': 476},
    'sutar_feni': {'standard_portion_desc': '1 serving', 'standard_portion_g': 50, 'calories_per_100g': 476},
    'unni_appam': {'standard_portion_desc': '1 piece', 'standard_portion_g': 55, 'calories_per_100g': 345}
}

# ---- Load YOLO model ----
try:
    model = YOLO(MODEL_PATH)
    logger.info(f"✓ Model loaded: {MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    model = None

# ---- Load class mapping ----
try:
    with open(CLASS_MAPPING_PATH) as f:
        class_mapping = json.load(f)
        class_to_id = class_mapping["class_to_id"]
        id_to_class = {int(k): v for k, v in class_mapping["id_to_class"].items()}
    logger.info(f"✓ Class mapping loaded: {len(class_to_id)} classes")
except Exception as e:
    logger.error(f"Failed to load class mapping: {str(e)}")
    id_to_class = {}

@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "OK",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", tags=["Inference"])
async def predict(file: UploadFile = File(...), conf: float = 0.5):
    """
    Accepts an image and returns detected food items along with
    portion size and calorie information (from FOOD_DATA lookup).
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        results = model.predict(source=image, conf=conf, verbose=False)
        result = results[0]

        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = id_to_class.get(class_id, f"class_{class_id}")

            # lookup calorie info
            food_info = FOOD_DATA.get(class_name, None)
            if food_info:
                portion = food_info["standard_portion_desc"]
                portion_g = food_info["standard_portion_g"]
                calories = round((portion_g * food_info["calories_per_100g"]) / 100, 2)
            else:
                portion = "Unknown"
                portion_g = None
                calories = None

            detections.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "portion_desc": portion,
                "portion_g": portion_g,
                "calories_estimate": calories
            })

        return {
            "image": file.filename,
            "num_detections": len(detections),
            "detections": detections,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

@app.get("/model_info", tags=["Info"])
def model_info():
    return {
        "model_path": MODEL_PATH,
        "num_classes": len(id_to_class),
        "classes": id_to_class
    }

@app.get("/", tags=["Info"])
def root():
    return {
        "name": "Food Detection API",
        "version": "1.1.0",
        "docs": "/docs",
        "redoc": "/redoc",
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")