install:
	pip install -r requirements.txt
ingest:
	python src/ingest.py
rag:
	python src/ingest.py && uvicorn src.app:app --reload
test:
	pytest tests/
