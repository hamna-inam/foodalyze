# Foodalyze
## Your pocket nutritionist, powered by AI.
A *YOLOv8-powered API* for detecting Indian food dishes and estimating their calorie content.

-----

## Architecture

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
```

## Dataset: [https://drive.google.com/file/d/1SOqkv7GBLbs\_f6AISFvczdh1rRS29\_AP/view?usp=sharing](https://drive.google.com/file/d/1SOqkv7GBLbs_f6AISFvczdh1rRS29_AP/view?usp=sharing)

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

Tracks and versions all trained YOLO models.

Tracking URI: file:///Users/bstar/Documents/Fall25/MLOps/MLFlow/mlruns

Experiment Name: *YOLOv8\_Indian\_Food\_Detection*
Registered Model: *Foodalyze\_YOLOv8\_Detector*

Trained for 30 epochs
Image size: 640×640

\<img width="2876" height="1434" alt="image" src="[https://github.com/user-attachments/assets/d09536d2-0d24-46f4-993c-ceaf6246b23e](https://github.com/user-attachments/assets/d09536d2-0d24-46f4-993c-ceaf6246b23e)" /\>

\<img width="1600" height="427" alt="image" src="[https://github.com/user-attachments/assets/92bda33a-2263-40c5-a082-cb47b7cafae3](https://github.com/user-attachments/assets/92bda33a-2263-40c5-a082-cb47b7cafae3)" /\>

\<img width="1600" height="472" alt="image" src="[https://github.com/user-attachments/assets/47dd52a2-8437-469a-9c35-a202b8c2abd3](https://github.com/user-attachments/assets/47dd52a2-8437-469a-9c35-a202b8c2abd3)" /\>

-----

### ✅ Evidently (Data Drift)

\<img width="1600" alt="image" src="[https://github.com/user-attachments/assets/ac783da0-6113-4441-9952-0458e560fc72](https://github.com/user-attachments/assets/ac783da0-6113-4441-9952-0458e560fc72)" /\>
\<img width="1600" alt="image" src="[https://github.com/user-attachments/assets/0e8299ac-8370-46df-9fe3-8b3e8ea6e1d0](https://github.com/user-attachments/assets/0e8299ac-8370-46df-9fe3-8b3e8ea6e1d0)" /\>

-----

### ✅ Prometheus + Grafana (API Metrics)

Prometheus scrapes live API metrics.
Grafana visualizes:

  * CPU usage
  * Memory usage
  * CPU temperature (simulated)

\<img width="1141" height="612" src="[https://github.com/user-attachments/assets/4448f63f-d787-466d-8cab-91468091bc84](https://github.com/user-attachments/assets/4448f63f-d787-466d-8cab-91468091bc84)" /\>
\<img width="1155" height="318" src="[https://github.com/user-attachments/assets/9d99bd73-b713-45e5-9665-8c12d3aa4031](https://github.com/user-attachments/assets/9d99bd73-b713-45e5-9665-8c12d3aa4031)" /\>
\<img width="1159" height="613" src="[https://github.com/user-attachments/assets/08bf807f-0f56-4850-b4cb-5276f2987d63](https://github.com/user-attachments/assets/08bf807f-0f56-4850-b4cb-5276f2987d63)" /\>

-----

## Tech Stack

  * *Backend:* FastAPI
  * *ML Framework:* PyTorch + YOLOv8
  * *Monitoring:* MLflow, Evidently, Prometheus, Grafana
  * *Containerization:* Docker & Docker Compose
  * *Cloud:* AWS EC2 + CloudWatch

-----

## ✅ Cloud Deployment (D9 Cloud Integration)

This project uses *two AWS cloud services*:

✅ *EC2* — hosts the inference API
✅ *CloudWatch* — monitors logs and system metrics

-----

## 1\. AWS Services Used

### ✅ 1.1 EC2 – Hosting the API

Runs Dockerized FastAPI YOLO inference server.

\<img width="705" height="146" src="[https://github.com/user-attachments/assets/43267c18-9a60-440b-b058-bf4aec6d6547](https://github.com/user-attachments/assets/43267c18-9a60-440b-b058-bf4aec6d6547)" /\>

-----

### ✅ 1.2 CloudWatch – System Monitoring

Collects:

  * CPU utilization
  * Memory usage
  * Disk usage
  * Logs

-----

## 2\. Deployment Architecture

\<img width="626" height="426" src="[https://github.com/user-attachments/assets/eef957d7-50c6-4ba5-ba52-8a05876d3be4](https://github.com/user-attachments/assets/eef957d7-50c6-4ba5-ba52-8a05876d3be4)" /\>

-----

## 3\. Reproducing the Setup

### ✅ 3.1 Launch EC2

1.  Ubuntu 22.04/24.04
2.  Open port *8000*

-----

### ✅ 3.2 Install Docker

```bash
sudo apt update
sudo apt install -y docker.io
sudo systemctl enable docker
sudo systemctl start docker
```

-----

### ✅ 3.3 Clone Repo

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd Foodalyze
```

-----

### ✅ 3.4 Build & Run Docker Container

```bash
sudo docker build -t foodalyze-api .
sudo docker run -d -p 8000:8000 foodalyze-api
sudo docker ps
```

-----

### ✅ 3.5 Access API

Open:
http://\<EC2-PUBLIC-IP\>:8000/docs

\<img width="1250" height="667" src="[https://github.com/user-attachments/assets/4c34b4fe-4d1d-4d3b-9914-72bb55d157a7](https://github.com/user-attachments/assets/4c34b4fe-4d1d-4d3b-9914-72bb55d157a7)" /\>

-----

### ✅ 3.6 Install & Configure CloudWatch Agent

```bash
# Download
wget https://s3.amazonaws.com/amazoncloudwatchagent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb

# Install
sudo dpkg -i amazon-cloudwatch-agent.deb

# Run config wizard
sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-config-wizard

# Start agent
sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
    -a fetch-config -m ec2 \
    -c file:/opt/aws/amazon-cloudwatch-agent/bin/config.json -s

# Check status
sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl -m ec2 -a status
```

\<img width="882" height="116" src="[https://github.com/user-attachments/assets/b71adf69-1342-45bc-82c9-e7b43a7e568e](https://github.com/user-attachments/assets/b71adf69-1342-45bc-82c9-e7b43a7e568e)" /\>

\<img width="1362" height="568" src="[https://github.com/user-attachments/assets/4a70efc4-258d-4d83-898c-a5e97a916108](https://github.com/user-attachments/assets/4a70efc4-258d-4d83-898c-a5e97a916108)" /\>

-----

## ✅ 4. ML Workflow Interaction With Cloud Services

### ✅ 4.1 Model Storage

The YOLO model (best.pt) is stored and loaded from inside the Docker container.

-----

### ✅ 4.2 Inference on EC2

1.  User uploads image
2.  FastAPI decodes using cv2
3.  YOLO model performs detection
4.  Calories added using lookup
5.  JSON returned

-----

## Note: Docker compose isnt complete (bonus path)

## License

This project is licensed under the *MIT License*.
See the [LICENSE](https://www.google.com/search?q=LICENSE) file for more details.

-----
