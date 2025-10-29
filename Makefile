# Use .PHONY to ensure these targets run even if a file with the same name exists
.PHONY: all dev lint test format docker run monitor-up monitor-down install

# --- Local Environment Setup ---

install:
	# Installs dependencies in a new virtual environment
	@echo "Creating virtual environment 'venv'..."
	python -m venv venv
	@echo "Virtual environment created. Activate with: source venv/bin/activate"
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements.txt
	@echo "✅ Dependencies installed."

# --- Main Development Targets ---

dev:
	# Runs the app in development mode with live-reloading
	# We use app:app because your file is named app.py
	uvicorn app:app --host 0.0.0.0 --port 8000 --reload

lint:
	# Runs the linters (for D4) - only checks for errors
	@echo "Checking formatting with black..."
	black --check .
	@echo "Linting with ruff..."
	ruff .

format:
	# Automatically fixes formatting and linting issues
	@echo "Fixing formatting with black..."
	black .
	@echo "Fixing linting issues with ruff..."
	ruff --fix .

test:
	# Runs tests with coverage (for D4)
	# Assumes tests will be configured to cover 'app.py'
	pytest --cov=app --cov-fail-under=80

# --- Docker Targets (D3) ---

docker:
	# Builds the production Docker image
	docker build -t foodalyze:latest .

run:
	# Runs the built Docker image locally
	docker run -p 8000:8000 foodalyze:latest

# --- Monitoring Targets (D5) ---

monitor-up:
	# Starts all services (app, mlflow, prometheus, grafana) in the background
	@echo "Starting monitoring stack (Prometheus, Grafana, MLflow)..."
	docker-compose up -d --build

monitor-down:
	# Stops and removes all running containers from docker-compose
	@echo "Stopping monitoring stack..."
	docker-compose down
