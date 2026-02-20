# Use a slim image to save disk space on EC2
FROM python:3.10-slim

# Set environment variables to optimize Python performance
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-cache \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create the local storage folder for model caching
RUN mkdir -p /app/model_cache && chmod 777 /app/model_cache

# Copy the application code
COPY app.py .

# Expose FastAPI port
EXPOSE 8000

# Run with a single worker to keep RAM usage stable on t2.small
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]