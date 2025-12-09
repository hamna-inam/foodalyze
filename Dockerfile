FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies needed for OpenCV + builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# -----------------------------------------------------------------------------

FROM python:3.11-slim

WORKDIR /app

# Install OpenCV runtime requirements
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ---- FIX: Create non-root user in this stage ----
RUN groupadd --system app && useradd --system --gid app app

# Copy Python deps
COPY --from=builder /install /usr/local

# Copy application files
COPY class_mapping.json .
COPY models/ models/
COPY src/ src/

# ---- FIX: chown works now because user exists ----
RUN chown -R app:app /app

RUN mkdir -p /home/app/.cache/huggingface
RUN chown -R app:app /home/app/.cache/huggingface

# Switch to app user
USER app

ENV YOLO_CONFIG_DIR=/tmp
ENV TRANSFORMERS_CACHE=/tmp
ENV MPLCONFIGDIR=/tmp
ENV HF_HOME=/tmp/huggingface
ENV TRANSFORMERS_CACHE=/tmp/huggingface

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
