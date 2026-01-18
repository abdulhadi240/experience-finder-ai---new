# ---- Base image ----
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Prevent Python from writing pyc files to disk & enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies (for building wheels & some libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc curl \
    && rm -rf /var/lib/apt/lists/*

# ---- Install dependencies ----
COPY requirements.txt .

# Upgrade pip and prefer precompiled binaries
RUN pip install --upgrade pip setuptools wheel \
 && pip install --prefer-binary --no-cache-dir -r requirements.txt

# ---- Copy application code ----
COPY . .

# Expose the port ECS/Terraform expect
EXPOSE 8080

# ---- Health check (optional but great for ECS) ----
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# ---- Run the FastAPI app ----
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
