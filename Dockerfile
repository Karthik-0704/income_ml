# Use an official, lightweight Python runtime
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first (this leverages Docker layer caching)
COPY requirements.txt .

# Install dependencies without storing cache to keep the image small
RUN pip install --no-cache-dir -r requirements.txt

# Copy the actual API script into the container
COPY app.py .

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the API using Uvicorn (Exec form allows graceful shutdowns)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
