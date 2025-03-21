# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p datasets .cache plots saved_model

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/saved_model
ENV PORT=8080

# Expose port (Cloud Run will use PORT environment variable)
EXPOSE ${PORT}

# Run the API with production settings
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"] 