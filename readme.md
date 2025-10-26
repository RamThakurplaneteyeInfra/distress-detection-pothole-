<<<<<<< HEAD
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
=======
# PotholeDetector

A real-time pothole detection web application using a Flask backend, a Segment-Anything Model (SAM) for image analysis, and a Leaflet-based map for visualization.

## Features

- **Real-time Detection**: Capture images from your device's camera or upload existing photos.
- **AI-Powered Analysis**: Uses the Segment-Anything Model (SAM) to automatically identify and segment potholes.
- **Detailed Reports**: Provides information on pothole severity, estimated area, depth, and detection confidence.
- **Live Updates**: New detections are broadcast to all connected clients in real-time using WebSockets (Socket.IO).
- **Interactive Map**: Visualizes all reported pothole locations on a live map.
- **PDF Export**: Generate and download a PDF report for any detected pothole.

## Local Setup and Installation

1.  **Clone the Repository**

    ```bash
    git clone <your-repository-url>
    cd distress-detection-pothole
    ```

2.  **Create and Activate a Virtual Environment**

    It's highly recommended to use a virtual environment to manage project dependencies.

    - **On macOS/Linux:**

      ```bash
      python3 -m venv venv
      source venv/bin/activate
      ```

    - **On Windows (Command Prompt):**

      ```bash
      python -m venv venv
      venv\Scripts\activate
      ```

    - **On Windows (PowerShell):**
      ```powershell
      python -m venv venv
      .\venv\Scripts\Activate.ps1
      ```

3.  **Install Dependencies**

    Install all the required packages from `requirements.txt`.

    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Application**

    The SAM model checkpoint (`sam_vit_b_01ec64.pth`) will be downloaded automatically on the first run if it's not found in the `data/models/` directory.

    #### For Development

    This method uses the built-in Flask development server, which is great for local testing.

    ```bash
    python app.py
    ```

    Navigate to `http://127.0.0.1:5000` in your web browser.

    #### For Production (with Gunicorn on Linux/macOS)

    For a more robust setup, use a production-grade WSGI server like Gunicorn. The project is already configured for it.

    ```bash
    gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:5000 "app:create_app()"
    ```

    > **Note on Windows:** Gunicorn is not supported on Windows. For a production-like server on Windows, you can use `waitress`:
    >
    > ```bash
    > pip install waitress
    > waitress-serve --host 0.0.0.0 --port 5000 --call "app:initialize_app" app:socketio
    > ```
>>>>>>> dcb6b30d9a6b43fc416fed284ed7bf3d7b7b2c40
