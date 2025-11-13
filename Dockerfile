# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (add OpenCV, model essentials)
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create uploads and model directories in /tmp for Cloud Run
RUN mkdir -p /tmp/uploads /tmp/models

# (Optional, not required if handled at runtime): Download SAM model to /tmp/models/
RUN curl -L -o /tmp/models/sam_vit_b_01ec64.pth "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" || echo "Model download failed, will download at runtime"

# Expose port (8080 for Cloud Run, will be overridden by platform)
EXPOSE 8080

# Set environment variables
ENV PORT=8080
ENV FLASK_ENV=production

# Use Gunicorn + eventlet (critical for Flask-SocketIO!)
# Use "app:socketio" as the entrypoint object if using Flask-SocketIO
CMD exec gunicorn --worker-class eventlet --bind :$PORT --timeout 0 app:socketio
