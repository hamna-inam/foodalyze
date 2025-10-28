
---

````markdown
#  Foodalyze
A **YOLOv8-powered API** for detecting Indian food dishes and estimating their calorie content.

---

##  Architecture

This diagram shows the complete MLOps workflow — from model training to a monitored API endpoint.

```mermaid
graph TD
    subgraph "MLOps & Monitoring (D5)"
        D[best.pt Model] -- registers --> M(MLflow Model Registry);
        E(FastAPI App) -- scrapes --> P(Prometheus);
        P -- datasource --> G(Grafana Dashboard);
        T(Training Data) --> EV(Evidently Report);
        V(Validation Data) --> EV;
    end

    subgraph "CI/CD Pipeline (D4)"
        A[Git Push/PR] --> B(GitHub Actions);
        B -- runs --> L(Lint);
        B -- runs --> TST(Test);
        TST -- on pass --> BLD(Build Docker Image);
        BLD -- push --> R(GHCR Registry);
        R --> DPL(Canary Deploy);
        DPL --> AT(Acceptance Tests);
    end

    subgraph "Inference API (D3, D7)"
        U[User Upload] --> E;
        E -- loads model from --> D;
        E --> J(JSON Response);
    end
````

---

##  Quick Start

Get the API running locally in **two simple steps** 👇

###  Clone the Repository

```bash
git clone https://github.com/alina1114/Foodalyze.git
cd Foodalyze
```

###  Build and Run the App

This command installs dependencies, formats code, and starts the development server:

```bash
make dev
```

After running this, open:
 [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Makefile Commands

Common commands are managed via the **Makefile** for consistency and ease of use.

| Command             | Description                                                                             |
| ------------------- | --------------------------------------------------------------------------------------- |
| `make install`      | Creates a Python virtual environment and installs dependencies from `requirements.txt`. |
| `make dev`          | Runs the FastAPI server in development mode with live-reloading.                        |
| `make lint`         | Checks for linting errors using `ruff` and `black`.                                     |
| `make format`       | Automatically fixes formatting and linting issues.                                      |
| `make test`         | Runs the `pytest` suite with coverage.                                                  |
| `make docker`       | Builds the production-ready Docker image.                                               |
| `make run`          | Runs the Docker image locally on `http://localhost:8000`.                               |
| `make monitor-up`   | Starts the full monitoring stack (App, MLflow, Prometheus, Grafana).                    |
| `make monitor-down` | Stops and removes monitoring containers.                                                |

---

##  API Documentation (D7)

FastAPI automatically generates documentation:

* **Swagger UI (Interactive):** [http://localhost:8000/docs](http://localhost:8000/docs)
* **ReDoc (Static):** [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

### Health Check

**Endpoint:** `GET /health`
Checks if the API is running and the model is loaded.

---

###  Predict Endpoint

Upload an image to get **food detections**, **bounding boxes**, and **calorie estimates**.

**Endpoint:** `POST /predict`
**Query Parameter:** `conf=0.5` (optional)
**Body:** `file` (image as `multipart/form-data`)

#### Example cURL Request

```bash
curl -X 'POST' \
  'http://localhost:8000/predict?conf=0.4' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@/path/to/your/image.jpg'
```

#### Example JSON Response

```json
{
  "image": "your_image.jpg",
  "num_detections": 1,
  "detections": [
    {
      "class_id": 12,
      "class_name": "chana_masala",
      "confidence": 0.9234,
      "bbox": {
        "x1": 150,
        "y1": 210,
        "x2": 450,
        "y2": 500
      },
      "portion_desc": "1 bowl",
      "portion_g": 240,
      "calories_estimate": 348
    }
  ],
  "timestamp": "2025-10-28T13:00:00.000000"
}
```

---

##  ML Workflow Monitoring (D5)

The project includes a **complete monitoring stack**, managed via `docker-compose`.

### MLflow (Model Registry)

Tracks and versions all trained models.

* **URL:** [http://localhost:5000](http://localhost:5000)
* **Registered Model:** *(Paste your MLflow model URL here)*

---

###  Evidently (Data Drift)

Monitors for **data drift** between training and validation datasets.

* **Dashboard:** [http://localhost:7000](http://localhost:7000)

---

###  Prometheus & Grafana (API Metrics)

* **Prometheus** scrapes live API metrics (requests, latency, errors).
* **Grafana** visualizes these metrics in real-time.

**Grafana URL:** [http://localhost:3000](http://localhost:3000)
**Login:** `admin / admin`

---

## ☁️ Cloud Deployment (Bonus D9)

The app can be deployed on **AWS** using **S3** and **EC2**.

###  AWS S3 (Model Storage)

**Why:**
S3 provides durable, scalable, and low-cost storage for the `best.pt` model file — decoupling model updates from Docker builds.

**Steps:**

1. Create an S3 bucket
2. Upload `best.pt`
3. Attach IAM role to EC2 with S3 read access

---

###  AWS EC2 (App Hosting)

**Why:**
EC2 offers a stable environment for hosting the Dockerized FastAPI app.

**Steps:**

```bash
sudo yum update -y
sudo amazon-linux-extras install docker -y
sudo service docker start
sudo usermod -a -G docker ec2-user
```

Then, pull and run the container:

```bash
docker run -d -p 80:8000 ghcr.io/alina1114/foodalyze:latest
```

 Your API will be live at:
`http://<your-ec2-public-ip>/docs`

---

##  Tech Stack

* **Backend:** FastAPI
* **ML Framework:** PyTorch + YOLOv8
* **Monitoring:** MLflow, Evidently, Prometheus, Grafana
* **Containerization:** Docker & Docker Compose
* **Cloud:** AWS EC2 + S3

---

## License

This project is licensed under the **MIT License**.
See the [LICENSE](LICENSE) file for more details.

```


```
