README.md 
# PotholeDetector

Real-time pothole detection web app with Flask backend and Leaflet map visualization.

## Features

- Capture camera images or upload photos
- Automatic pothole detection using SAM model
- Display detection severity, area, and confidence
- Real-time updates via Socket.IO
- Map visualization of pothole locations

## Setup

1. Clone the repo:

```bash
git clone https://github.com/yourusername/pothole-detector.git
cd pothole-detector


Create virtual environment:

python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows


Install dependencies:

pip install -r requirements.txt


Run the app:

python app.py


The SAM model will be downloaded automatically if missing.
