# ════════════════════════════════════════════════════
# Stage 1: Builder
# ════════════════════════════════════════════════════
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps needed to compile some Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ════════════════════════════════════════════════════
# Stage 2: Production
# ════════════════════════════════════════════════════
FROM python:3.11-slim AS production

WORKDIR /app

# Only runtime system deps (no gcc, no g++)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy source code
COPY src/ ./src/

# Create necessary directories
RUN mkdir -p /app/data/processed /app/output

# Expose FastAPI port
EXPOSE 8000

# Healthcheck — tells docker-compose when app is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run FastAPI app
CMD ["uvicorn", "src.orchestrator.main:app", "--host", "0.0.0.0", "--port", "8000"]
