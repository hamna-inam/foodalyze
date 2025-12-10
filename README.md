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
# Foodalyze

## Your pocket nutritionist, powered by AI.
A dual-engine system: **YOLOv8** for food detection and **DeepSeek-LLM** for nutritional RAG analysis.

-----

## 🏗️ System Architecture

This project consists of two main pipelines: Computer Vision (CV) for detection and Large Language Models (LLM) for RAG-based Q&A.

### 1. Computer Vision & MLOps Pipeline (YOLOv8)
This diagram shows the complete MLOps workflow — from model training to a monitored API endpoint.

```mermaid
graph TD
    subgraph "MLOps & Monitoring (D5)"
        D[best.pt Model] -- registers --> M(MLflow Model Registry)
        E(FastAPI App) -- scrapes --> P(Prometheus)
        P -- datasource --> G(Grafana Dashboard)
        T(Training Data) --> EV(Evidently Report)
        V(Validation Data) --> EV
    end
  
    subgraph "CI/CD Pipeline (D4)"
        A[Git Push/PR] --> B(GitHub Actions)
        B -- runs --> L(Lint)
        B -- runs --> TST(Test)
        TST -- on pass --> BLD(Build Docker Image)
        BLD -- push --> R(GHCR Registry)
        R --> DPL(Canary Deploy)
        DPL --> AT(Acceptance Tests)
    end
  
    subgraph "Inference API (D3, D7)"
        U[User Upload] --> E
        E -- loads model from --> D
        E --> J(JSON Response)
    end
````

### 2\. LLM RAG Pipeline (Milestone 2)

This diagram illustrates the retrieval, augmentation, and generation flow for the AI Nutritionist chatbot.

```mermaid
graph TD
    %% Nodes
    User([👤 User])
    UI[🖥️ Frontend Streamlit App]
    API[⚙️ Backend API FastAPI]
    VDB[(🗄️ Vector DB FAISS Index)]
    LLM[🧠 LLM Model DeepSeek-7B-Chat]
    
    %% Flow
    User -->|"1. Asks: Is Samosa healthy?"| UI
    UI -->|"2. Sends JSON Request"| API
    
    subgraph "RAG Pipeline"
        API -->|"3. Query Embeddings"| VDB
        VDB -->|"4. Retrieve Context (Nutrition Data)"| API
        API -->|"5. Construct Prompt (Context + Query)"| LLM
        LLM -->|"6. Generate Answer"| API
    end
    
    API -->|"7. JSON Response"| UI
    UI -->|"8. Display Answer"| User

    %% Styling
    style VDB fill:#f9f,stroke:#333,stroke-width:2px
    style LLM fill:#bbf,stroke:#333,stroke-width:2px
    style API fill:#dfd,stroke:#333,stroke-width:2px
```

<img width="1289" height="435" alt="image" src="https://github.com/user-attachments/assets/cd28f5d1-ada4-4e93-a313-ac4737d4dd0e" />


## Dataset

**Access the Dataset Here:** [Google Drive Link](https://drive.google.com/file/d/1SOqkv7GBLbs_f6AISFvczdh1rRS29_AP/view?usp=sharing)

```
```
### 3.Data Flow & Ingestion (Milestone 2)
How raw nutrition data is processed and stored for retrieval.

```mermaid
flowchart LR
    %% Data Sources
    CSV["📄 nutrition.csv"]
    JSON["📄 swaps.json"]
    
    %% Processes
    Ingest["⚙️ Ingestion Script<br/>(src/ingest.py)"]
    Embed["🔠 Encoder Model<br/>(all-MiniLM-L6-v2)"]
    
    %% Storage
    VDB[("🗄️ Vector Store<br/>(FAISS Local Index)")]
    Cloud["☁️ Cloud Storage<br/>(S3 / Artifacts)"]
    
    %% Connections
    CSV & JSON -->|Read Raw Data| Ingest
    Ingest -->|Clean & Chunk Text| Embed
    Embed -->|Generate Vectors| VDB
    
    %% Cloud Sync (D7 Requirement)
    VDB -.->|Backup/Sync| Cloud
    
    %% Styling
    style Ingest fill:#ff9,stroke:#333
    style VDB fill:#f9f,stroke:#333
    style Cloud fill:#eee,stroke:#333,stroke-dasharray: 5 5
```
    

## Quick Start

Get the API running locally in *two simple steps*.

### Clone the Repository

```bash
git clone https://github.com/alina1114/Foodalyze.git
cd Foodalyze
```

### Build and Run the App

This command installs dependencies, formats code, and starts the development server:

```bash
make dev
```

Then open:
👉 [http://localhost:8000/docs](https://www.google.com/search?q=http://localhost:8000/docs)

-----

## Makefile Commands
## Note: Docker compose isnt complete (bonus path)

| Command | Description |
| :--- | :--- |
| make install | Creates a Python virtual environment and installs dependencies. |
| make dev | Starts the FastAPI server with live reload. |
| make lint | Runs lint checks (ruff, black). |
| make format | Auto-formats code. |
| make test | Runs tests with pytest. |
| make docker | Builds Docker image. |
| make run | Runs Docker container locally. |
| make monitor-up | Starts MLflow, Prometheus, Grafana. |
| make monitor-down | Stops monitoring stack. |

-----
🛡️ Guardrails & Safety (Milestone 2 - D3) 
We utilize a custom Policy Engine (src/guardrails.py) to enforce Responsible AI guidelines. This ensures the system remains safe, secure, and helpful.

Protection Layers
Input Validation (Security):

PII Filter: Blocks emails, phone numbers, SSNs, and credit card patterns to prevent data leakage.

Prompt Injection: Detects and blocks adversarial attacks like "Ignore previous instructions" or "System Override".

Output Moderation (Safety):

Toxicity Filter: Scans the generated response for harmful keywords (violence, hate speech) before showing it to the user.

✅ Verification Results
We performed an automated stress test against 12 different attack vectors (including PII leaks, DAN jailbreaks, and roleplay attacks).

🏆 Result: 12/12 Passed (100% Block Rate) All malicious attempts were successfully intercepted by the guardrail middleware layer (HTTP 400 Bad Request).

<img width="552" height="810" alt="image" src="https://github.com/user-attachments/assets/17b95f6f-17e1-431b-8b4f-39dfb951f4df" />

![WhatsApp Image 2025-12-07 at 9 22 34 PM](https://github.com/user-attachments/assets/0b561b09-12f6-4437-a08d-9789cc6b57ed)



## API Documentation (D7)

FastAPI automatically generates documentation:

  * *Swagger UI:* [http://localhost:8000/docs](https://www.google.com/search?q=http://localhost:8000/docs)
  * *ReDoc:* [http://localhost:8000/redoc](https://www.google.com/search?q=http://localhost:8000/redoc)

-----

### ✅ Health Check (GET /health)

Verifies API status and model load.

-----

### ✅ Predict Endpoint (POST /predict)

Upload an image → receive \*detections, \*\*bounding boxes, and \**calorie estimates*.
<img width="1280" height="638" alt="image" src="https://github.com/user-attachments/assets/297a06bf-4ad4-4e17-be6e-328d81a5ea86" />


#### Example cURL

```bash
curl -X 'POST' \
  'http://localhost:8000/predict?conf=0.4' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@/path/to/image.jpg'
```

#### Example Response

```json
{
  "image": "your_image.jpg",
  "num_detections": 1,
  "detections": [
    {
      "class_id": 12,
      "class_name": "chana_masala",
      "confidence": 0.9234,
      "bbox": {"x1": 150, "y1": 210, "x2": 450, "y2": 500},
      "portion_desc": "1 bowl",
      "portion_g": 240,
      "calories_estimate": 348
    }
  ],
  "timestamp": "2025-10-28T13:00:00.000000"
}
```

-----

## ML Workflow Monitoring (D5)

### ✅ MLflow (Model Registry)

Tracked YOLO runs locally in Milestone 1 then moved the runs (only) to the server in Milestone 2.
Endpoint: 

Tracks and versions all trained YOLO models.

Tracking URI: file:///Users/bstar/Documents/Fall25/MLOps/MLFlow/mlruns

Experiment Name: *YOLOv8\_Indian\_Food\_Detection*
Registered Model: *Foodalyze\_YOLOv8\_Detector*

Trained for 30 epochs

<img width="1280" height="638" alt="image" src="https://github.com/user-attachments/assets/34076a1f-47c7-4fcf-a35f-35dd1b25d8b7" />

<img width="1280" height="638" alt="image" src="https://github.com/user-attachments/assets/607c9e6c-fc5d-409c-b97f-20c52d5a1fa9" />

<img width="1600" height="427" alt="image" src="https://github.com/user-attachments/assets/55e16571-cf29-4041-bcab-1a0c045bb69d" />

<img width="1600" height="472" alt="image" src="https://github.com/user-attachments/assets/9f032fd8-e5d2-4eab-933e-18672e3840f8" />




-----

### ✅ Evidently (Data Drift)

YOLO Drift (Milestone 1):

<img width="1280" height="606" alt="image" src="https://github.com/user-attachments/assets/5116ae98-6d4e-4c8d-a333-fe0ce4c79398" />
<img width="1280" height="643" alt="image" src="https://github.com/user-attachments/assets/34c674c5-3367-43c6-bb61-f754790e0daf" />


\<img width="1600" alt="image" src="[https://github.com/user-attachments/assets/0e8299ac-8370-46df-9fe3-8b3e8ea6e1d0](https://github.com/user-attachments/assets/0e8299ac-8370-46df-9fe3-8b3e8ea6e1d0)" /\>

### ✅ Evidently (Retrieval Corpus Drift)

The faiss index is static (indices can't be added incrementally so drift is 0)

<img width="1159" height="590" alt="image" src="https://github.com/user-attachments/assets/c782fa79-152f-4b0e-8488-3a6588351a76" />


-----

### ✅ Prometheus + Grafana (API Metrics) - Milestone 1

Link to Dashboards:
http://13.61.104.51:3000/dashboards


Prometheus scrapes live API metrics.
Grafana visualizes:

  * CPU usage
    <img width="2880" height="1800" alt="image" src="https://github.com/user-attachments/assets/94e326b3-36f1-46ed-a5c9-3b30d231ea8c" />

  * Memory usage

    <img width="2880" height="1800" alt="image" src="https://github.com/user-attachments/assets/ad4ea9b1-f115-4bea-9c6d-7687cd4a1856" />

  * CPU temperature (simulated)
    <img width="2880" height="1800" alt="image" src="https://github.com/user-attachments/assets/3e822a2b-f2a0-48dc-9974-c33950ce79ca" />

-----
## Prometheus + Grafana (API Metrics) - Milestone 2

##  *1. LLM Request Latency (P95)*

This graph shows the *95th percentile latency* of all LLM inference requests over time.

* Uses: histogram_quantile(0.95, rate(llm_request_duration_seconds_bucket[$__rate_interval]))
* Purpose: Tracks how long the slowest 5% of requests take.
* Interpretation:

  * Stable, low values indicate good LLM performance.
  * Spikes may mean CPU overload, large prompts, or model cold starts.

<img width="1297" height="524" alt="image" src="https://github.com/user-attachments/assets/dae69928-d05a-4e3e-8a8c-e4736fa19ed0" />


## *2. Guardrail Violations*

Displays the total count of *blocked or flagged responses* by the system’s safety guardrails.

* Metric: guardrail_violations_total
* Purpose: Measures how often the model produces unsafe/toxic/unsupported outputs.
* Interpretation:

  * A rising trend suggests more harmful inputs or weaker filters.
  * Helps verify safety systems are working correctly.
  <img width="1314" height="558" alt="image" src="https://github.com/user-attachments/assets/5af8e83b-6b8a-4b75-849b-28de38efc91b" />





    
## 🔡 *3. LLM Token Usage*

Tracks total *input* and *output* tokens processed by the model over time.

* Metric: llm_token_usage_total{type="input"}
* Metric: llm_token_usage_total{type="output"}
* Purpose: Understand model load, cost, and usage patterns.
* Interpretation:

  * Growth in input tokens means larger user queries.
  * Growth in output tokens indicates longer/generated answers.
 
     <img width="1309" height="531" alt="image" src="https://github.com/user-attachments/assets/5349d4a8-028b-4091-8060-9edc7d61874e" />

 

## 🖼️ *4. YOLO Prediction Count*

Shows how many YOLO detections were processed by the API.

* Metric: yolo_inference_duration_seconds_count
* Purpose: Monitor computer vision workload.
* Interpretation:

  * Spikes = increased image uploads.
  * Constant low activity means fewer classification requests.

<img width="1317" height="569" alt="image" src="https://github.com/user-attachments/assets/87f3c2ff-c1f5-4e1c-a84c-b12cba4e1b1b" />

## *5. YOLO Inference Latency (P95)*

Shows the *95th percentile latency* of YOLO detections.

* Uses: histogram_quantile(0.95, rate(yolo_inference_duration_seconds_bucket[$__rate_interval]))
* Purpose: Measure performance of image inference.
* Interpretation:

  * Usually low because YOLO is fast.
  * Spikes indicate CPU saturation or large images.
<img width="1672" height="838" alt="image" src="https://github.com/user-attachments/assets/be1f4249-bf4c-4811-bd8e-9ba4ec32a82a" />


## Tech Stack

  * *Backend:* FastAPI
  * *ML Framework:* PyTorch + YOLOv8
  * *Monitoring:* MLflow, Evidently, Prometheus, Grafana
  * *Containerization:* Docker & Docker Compose
  * *Cloud:* AWS EC2 + CloudWatch

-----

## 
-----

## ☁️ Cloud Integration

For Milestone 2, we integrated **two AWS cloud services**:  
**Amazon EC2** for hosting the inference API and **Amazon S3** for storing model artifacts and FAISS indexes.  
CloudWatch was also used for monitoring API and system metrics.

---

### 🔹 **1. Amazon S3 — Artifact & Vector Store Storage**

We use S3 to store:

- `best.pt` (YOLO model weights)  
- `class_mapping.json`  
- `index.faiss` (RAG vector index)  
- `index.pkl` (metadata)

These artifacts are automatically loaded by the ingestion pipeline when deploying the RAG system.

<img width="1338" height="428" alt="image" src="https://github.com/user-attachments/assets/cca3c344-602c-42d7-b0e3-fcca737f3551" />

---

### 🔹 **2. Amazon EC2 — Hosting the Foodalyze API**

The API (FastAPI + YOLOv8 + RAG) is deployed as a Docker container on AWS EC2.

- Public IP: `13.61.104.51`
- Open ports: 8000 (API), 3000 (Grafana), 8501 (UI)
- Deployed via Docker:
  `sudo docker run -d -p 8000:8000 foodalyze-api`

<img width="498" height="291" alt="image" src="https://github.com/user-attachments/assets/2fe86e62-0a01-49b0-8afe-8691e1f17d44" />


---

### 🔹 **3. Amazon CloudWatch — Monitoring the EC2 Instance**

CloudWatch tracks:

- CPU utilization (shows inference load)
- System-level metrics  
- Docker-level activity when CPU spikes occur during model inference


<img width="1362" height="568" alt="image" src="https://github.com/user-attachments/assets/652ea5ae-619c-463a-91a8-2525eadafa77" />


