# Stage 1: Builder - Install dependencies securely
FROM python:3.11-slim as builder

WORKDIR /app

# Create a non-root user for security
RUN groupadd --system app && useradd --system --gid app app

# Install system build tools for packages like box2d-py and twofish
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    swig \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt


# Stage 2: Final Image - Copy only what's needed
FROM python:3.11-slim

# Install only runtime libraries needed by OpenCV and curl (for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /app

# Copy non-root user from builder
COPY --from=builder /etc/passwd /etc/passwd
COPY --from=builder /etc/group /etc/group

# Copy installed dependencies from builder
# This part can be heavy, but cleaning apt caches above helps prevent disk overflow
COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy application files
COPY app.py .
COPY class_mapping.json .
COPY models/ models/

# Change ownership and run as non-root user
RUN chown -R app:app /app
USER app

# Healthcheck (D3)
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Expose port and run application
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
