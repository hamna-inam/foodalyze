# 🍛 Healthy Indian Food AI (LLMOps Milestone 2)

## Project Overview
This is a RAG (Retrieval-Augmented Generation) system that acts as an AI Nutritionist for Indian Cuisine. 
It uses **DeepSeek-7B** (or Phi-3) to answer questions based on a curated dataset of nutrition facts and healthy swaps.

## 📂 Structure
- `src/`: Source code for the API (`app.py`), Ingestion (`ingest.py`), and Frontend (`ui.py`).
- `data/`: The dataset (`csv`, `json`) and evaluation data (`eval.jsonl`).
- `experiments/`: Prompt engineering reports and evaluation scripts (D1).
- `faiss_index/`: The pre-built Vector Database.

## 🚀 How to Run
1. **Install:** `make install`
2. **Run RAG API:** `make rag`
3. **Run Frontend:** `streamlit run src/ui.py`
