# Contributing

## 👥 Team Members
This project was a collaborative effort by:

| Name | ERP ID |
| :--- | :--- |
| Alina | 26555 |
| Hamna | 27113|
| Rafsha | 26639 |
| Zara | 26928 |


---

## 📊 Task Allocation
Tasks were divided based on the Milestone 1 deliverables.

| Member | Category | Tasks Completed |
| :--- | :--- | :--- |
<<<<<<< HEAD
| **Alina** | CI/CD & Data | <ul><li>Finalized, tested, and debugged the `ci.yml` workflow (D4).</li><li>Trained the final production model (`best.pt`)</li> <li>Set up DVC for data versioning.</li><li>Created the initial repository and project structure.</li><li>Modified and debugged `app.py`,`requirements.txt`, `dockerfile` for CI/CD pipeine .</li><li>Ran initial YOLO model tests.</li></ul> |
| **Hamna** | Monitoring (D5) & Data | <ul><li>Completed the entire monitoring stack (D5).</li><li>Set up and configured Prometheus and Grafana.</li><li>Generated the Evidently data drift report.</li><li>Configured the MLflow service for experiment tracking.</li><li>Prepared the dataset (created bounding boxes for YOLO).</li></ul> |
=======
| **Alina** | CI/CD & Data | <ul><li>Finalized, tested, and debugged the ci.yml workflow (D4).</li><li>Trained the final production model (best.pt).</li><li>Set up DVC for data versioning.</li><li>Created the initial repository and project structure.</li><li>Modified and debugged app.py, requirements.txt, and Dockerfile for the CI/CD pipeline.</li><li>Implemented mock dependencies in the API tests to improve test coverage (achieved 85%).</li><li>Performed initial dataset formatting and YOLO label corrections to ensure class files followed YOLO’s required labeling structure.</li><li>Ran initial YOLO model tests.</li></ul> |
| **Hamna** | Monitoring (D5) & Data | <ul><li>Completed the entire monitoring stack (D5).</li><li>Set up and configured Prometheus and Grafana.</li><li>Generated the Evidently data drift report.</li><li>Configured the MLflow service for experiment tracking.</li><li>Prepared the dataset (created bounding boxes and text labels for YOLO).</li> <li>Finalized the dataset with train, test and valid splits</li></ul> |
>>>>>>> ae734ebe3fa43000b7c816f49e3784141643e50c
| **Rafsha**| API, Infra, & Testing | <ul><li>Wrote the `Dockerfile` and `docker-compose.yml` (D3).</li><li>Wrote the `test_api.py` unit tests (D4b).</li><li>Set up `pre-commit` hooks (D6).</li><li>Modified `app.py` to comply with docker and made the calorie lookup function for prediction .</li><li>Created the initial `ci.yml` file and did some debugging commits.</li><li>Wrote the `README.md` (D1) and `CONTRIBUTING.md` (D2).</li></ul> |
| **Zara** | ML, API, & Cloud | <ul><li>Implemented the cloud integration (D9).</li><li> Configured Ec2 and cloudwatch (D9).</li><li>Wrote the API documentation (D7) for the `README.md`.</li><li>Logged initial experiments to MLflow (D5).</li><li>Ran initial YOLO model tests.</li></ul> |

---

## 🌿 Branch Naming Convention
We follow this convention for all branches to keep the repository clean:

* **`feat/...`**: For adding a new feature (e.g., `feat/add-prediction-endpoint`)
* **`fix/...`**: For bug fixes (e.g., `fix/docker-healthcheck-fail`)
* **`infra/...`**: For infrastructure and CI/CD changes (e.g., `infra/setup-prometheus`)
* **`docs/...`**: For documentation updates (e.g
