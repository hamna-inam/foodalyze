#!/usr/bin/env python3
"""
Evidently Data Drift Detection for Foodalyze RAG Retrieval Corpus
"""

import os
import json
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import faiss

# Evidently
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# FastAPI
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================
FAISS_INDEX_PATH = "./faiss_index"
DRIFT_REPORT_PATH = "./drift_report.html"
DRIFT_METRICS_PATH = "./drift_metrics.json"
DRIFT_THRESHOLD = 0.1

# ============================================================================
# LOAD FAISS INDEX
# ============================================================================
def load_faiss_index():
    """Load FAISS index directly"""
    try:
        index_file = os.path.join(FAISS_INDEX_PATH, "index.faiss")
        if os.path.exists(index_file):
            index = faiss.read_index(index_file)
            num_docs = index.ntotal
            logger.info(f"✅ FAISS index loaded: {num_docs} documents")
            return index
        else:
            logger.error(f"❌ Index file not found: {index_file}")
            return None
    except Exception as e:
        logger.error(f"❌ Failed to load FAISS: {e}")
        return None

faiss_index = load_faiss_index()

# ============================================================================
# DRIFT DETECTION FUNCTIONS
# ============================================================================
def get_num_documents():
    """Get number of documents"""
    if faiss_index is None:
        return 0
    return faiss_index.ntotal

def detect_drift():
    """Calculate drift score"""
    if faiss_index is None:
        return {"error": "FAISS not loaded", "drift_detected": False}
    
    try:
        num_docs = faiss_index.ntotal
        
        # Reference: first 100 documents
        ref_embeddings = faiss_index.reconstruct_n(0, min(100, num_docs))
        
        # Current: all documents
        curr_embeddings = faiss_index.reconstruct_n(0, min(500, num_docs))
        
        # L2 distance
        drift_score = float(np.mean(np.linalg.norm(
            curr_embeddings[:len(ref_embeddings)] - ref_embeddings,
            axis=1
        )))
        
        return {
            "drift_score": round(drift_score, 4),
            "drift_threshold": DRIFT_THRESHOLD,
            "drift_detected": drift_score > DRIFT_THRESHOLD,
            "num_documents": num_docs,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error in drift detection: {e}")
        return {"error": str(e), "drift_detected": False}

def generate_evidently_report():
    """Generate Evidently drift report"""
    if faiss_index is None:
        return {"error": "FAISS not loaded"}
    
    try:
        num_docs = faiss_index.ntotal
        
        # Get reference (first 100) and current (first 500)
        num_reference = min(100, num_docs)
        num_current = min(500, num_docs)
        
        ref_embeddings = faiss_index.reconstruct_n(0, num_reference)
        curr_embeddings = faiss_index.reconstruct_n(0, num_current)
        
        # Convert to DataFrames
        ref_df = pd.DataFrame(ref_embeddings)
        ref_df.columns = [f"dim_{i}" for i in range(ref_embeddings.shape[1])]
        
        curr_df = pd.DataFrame(curr_embeddings)
        curr_df.columns = [f"dim_{i}" for i in range(curr_embeddings.shape[1])]
        
        logger.info("📊 Generating Evidently report...")
        logger.info(f"  Reference: {num_reference} documents")
        logger.info(f"  Current: {num_current} documents")
        
        # Generate Evidently report
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref_df, current_data=curr_df)
        
        # Save HTML report
        report.save_html(DRIFT_REPORT_PATH)
        logger.info(f"✅ Report saved: {DRIFT_REPORT_PATH}")
        
        return {
            "report_path": DRIFT_REPORT_PATH,
            "reference_docs": num_reference,
            "current_docs": num_current,
            "timestamp": datetime.now().isoformat(),
            "status": "OK"
        }
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        return {"error": str(e)}

# ============================================================================
# FASTAPI APP
# ============================================================================
app = FastAPI(
    title="Foodalyze Drift Monitor",
    description="Monitor data drift in RAG retrieval corpus using Evidently",
    version="1.0.0"
)

@app.get("/")
def root():
    return {
        "service": "Foodalyze Drift Monitor",
        "endpoints": {
            "health": "/health",
            "drift": "/drift",
            "report": "/report",
            "generate": "POST /report/generate"
        }
    }

@app.get("/health")
def health():
    """Health check"""
    num_docs = get_num_documents()
    return {
        "status": "OK",
        "faiss_loaded": faiss_index is not None,
        "num_documents": num_docs,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/drift")
def get_drift_metrics():
    """Get drift metrics"""
    return detect_drift()

@app.post("/report/generate")
def generate_report():
    """Generate Evidently drift report"""
    logger.info("🔄 Generating drift report...")
    result = generate_evidently_report()
    
    if "error" not in result:
        # Save metrics
        drift_metrics = detect_drift()
        with open(DRIFT_METRICS_PATH, 'w') as f:
            json.dump(drift_metrics, f, indent=2)
        logger.info("✅ Drift metrics saved")
    
    return result

@app.get("/report")
def get_report():
    """Serve Evidently HTML report"""
    if os.path.exists(DRIFT_REPORT_PATH):
        return FileResponse(DRIFT_REPORT_PATH, media_type="text/html")
    return JSONResponse(
        {"error": "Report not generated. POST /report/generate first."},
        status_code=404
    )

@app.get("/metrics")
def get_metrics_json():
    """Get drift metrics as JSON"""
    if os.path.exists(DRIFT_METRICS_PATH):
        with open(DRIFT_METRICS_PATH, 'r') as f:
            return json.load(f)
    return {"error": "Metrics not available"}

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    if faiss_index is None:
        logger.warning(f"⚠️ FAISS index not found at {FAISS_INDEX_PATH}")
        logger.warning("Drift detection will not work until FAISS is loaded")
    else:
        logger.info(f"✅ FAISS ready with {get_num_documents()} documents")
    
    logger.info("🚀 Drift Monitor starting on http://0.0.0.0:7009")
    uvicorn.run(app, host="0.0.0.0", port=7009, log_level="info")
