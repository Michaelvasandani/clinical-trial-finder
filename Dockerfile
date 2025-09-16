# Multi-stage Dockerfile for Clinical Trial Finder

FROM python:3.11-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data /app/embeddings /app/conversations /app/logs

# FastAPI Backend Stage
FROM base as api
EXPOSE 8000
CMD ["python", "chat_api.py"]

# Streamlit Frontend Stage
FROM base as streamlit
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Default stage (API)
FROM api