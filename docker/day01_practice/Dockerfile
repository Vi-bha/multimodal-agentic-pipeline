FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first
# Docker caches this layer — if requirements.txt doesn't change,
# this step is skipped on rebuild = much faster builds
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app.py .

# Create output directory
RUN mkdir -p /app/output

# Command to run when container starts
CMD ["python", "app.py"]
