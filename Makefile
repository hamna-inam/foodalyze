install:
<<<<<<< HEAD
	pip install -r requirements.txt
ingest:
	python src/ingest.py
rag:
	python src/ingest.py && uvicorn src.app:app --reload
=======
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
	uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload

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

>>>>>>> 713448b (Milestone 2 Integrated)
test:
	pytest tests/
