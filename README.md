#  Foodalyze
A **YOLOv8-powered API** for detecting Indian food dishes and estimating their calorie content.


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

Get the API running locally in **two simple steps** 

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
![76b0b36d-4935-4ef0-a4e4-e58e225fcc33](https://github.com/user-attachments/assets/acf2c684-dfda-4d0e-aad5-65e0ec85cd89)

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

Tracks and versions all trained models, including logged parameters, metrics, and artifacts.

**Tracking URI**: file:///Users/bstar/Documents/Fall25/MLOps/MLFlow/mlruns

URL: http://localhost:5000

**Experiment Name**: YOLOv8_Indian_Food_Detection

**Registered Model**: Foodalyze_YOLOv8_Detector

Description: The model was logged and registered locally via MLflow using the YOLOv8 architecture.

Trained for 30 epochs

Image size: 640×640

<img width="2876" height="1434" alt="image" src="https://github.com/user-attachments/assets/d09536d2-0d24-46f4-993c-ceaf6246b23e" />

<img width="1600" height="427" alt="image" src="https://github.com/user-attachments/assets/92bda33a-2263-40c5-a082-cb47b7cafae3" />

<img width="1600" height="472" alt="image" src="https://github.com/user-attachments/assets/47dd52a2-8437-469a-9c35-a202b8c2abd3" />


---

###  Evidently (Data Drift)

Monitors for **data drift** between training and validation datasets.

•⁠  ⁠URL (local): http://localhost:7001  (On Mac, port 7000 is already taken by ControlCe, which is part of Control Center / AFS (Apple File System service)
•⁠  ⁠Report generated on held-out test set

* **Dashboard:** 
![1b695b93-1acc-4e06-ba9a-905c2fab8699](https://github.com/user-attachments/assets/ac783da0-6113-4441-9952-0458e560fc72)
![17fad28c-fbdf-4ab1-ba37-5fefd86cfca9](https://github.com/user-attachments/assets/0e8299ac-8370-46df-9fe3-8b3e8ea6e1d0)

---

###  Prometheus & Grafana (API Metrics)

**Prometheus URL**: http://localhost:9090

**Grafana URL**: http://localhost:3000

Metrics collected:
•⁠  ⁠CPU usage (%)
•⁠  ⁠Memory usage (%)
•⁠  ⁠CPU temperature (simulated)

<img width="1141" height="612" alt="image" src="https://github.com/user-attachments/assets/4448f63f-d787-466d-8cab-91468091bc84" />

<img width="1155" height="318" alt="image" src="https://github.com/user-attachments/assets/9d99bd73-b713-45e5-9665-8c12d3aa4031" />

<img width="1159" height="613" alt="image" src="https://github.com/user-attachments/assets/08bf807f-0f56-4850-b4cb-5276f2987d63" />

<img width="1159" height="613" alt="image" src="https://github.com/user-attachments/assets/4b284291-7ed6-46a4-9a6c-68ce515da7e1" />


*Note on Metrics:*

While the assignment requested GPU utilization metrics, this setup collects CPU metrics instead (CPU usage %, memory usage %, and simulated CPU temperature).

*Reason:*  
The local machine (MacBook) used for this project does not have a dedicated GPU accessible for monitoring by Python/Prometheus. Additionally, Kaggle/Colab GPU hours for this task had expired, preventing GPU-based experiments.  

Collecting CPU metrics demonstrates the same Prometheus + Grafana monitoring workflow, including real-time scraping and visualization, which fulfills the monitoring and MLOps objectives.  

This workflow can be scaled to GPU metrics in environments with available GPUs (e.g., cloud GPU instances or Kaggle with active GPU credits).

---

## Cloud Deployment (Bonus D9)

The app can be deployed on **AWS** and **EC2**.

![98ba1226-6a89-47a2-8a19-c3812f4ea977](https://github.com/user-attachments/assets/bf252fde-e8b6-459b-8651-570660c5a083)

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
* **Cloud:** AWS EC2 + cloudwatch

---

## License

This project is licensed under the **MIT License**.
See the [LICENSE](LICENSE) file for more details.

```


```
