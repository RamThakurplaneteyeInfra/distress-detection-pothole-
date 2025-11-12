# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl git ffmpeg libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code
COPY . .

# Create uploads directory
RUN mkdir -p uploads

# (Optional) Pre-download the SAM model weights
# If download fails, your Flask app will handle it dynamically at runtime
RUN curl -L -o sam_vit_b_01ec64.pth \
    "https://huggingface.co/AkhileshYR/sam-vit-b-model/resolve/main/sam_vit_b_01ec64.pth" \
    || echo "⚠️ Model download failed - will download at runtime"

# Expose port (Cloud Run uses 8080)
EXPOSE 8080

# Environment variables
ENV PORT=8080
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1
ENV GUNICORN_CMD_ARGS="--workers=1 --threads=8 --timeout=0"

# Start the app via Gunicorn + Eventlet (for Flask-SocketIO)
CMD exec gunicorn --bind :$PORT --worker-class eventlet app:app
