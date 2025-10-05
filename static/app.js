# app.py - Full Windows-Compatible Version
from flask import Flask, request, jsonify, render_template, send_file
from flask_socketio import SocketIO
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
import io
import os
import logging
import sqlite3
from datetime import datetime
from flask_cors import CORS
import folium

# ------------------------
# Logging configuration
# ------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------
# Flask & SocketIO setup
# ------------------------
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# ------------------------
# Configuration
# ------------------------
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DATABASE'] = 'potholes.db'
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ------------------------
# Global variables
# ------------------------
predictor = None
sam_loaded = False

# ------------------------
# SAM Model Initialization
# ------------------------
def init_sam():
    """Initialize SAM model"""
    global predictor, sam_loaded
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        sam_checkpoint = "sam_vit_b_01ec64.pth"
        if not os.path.exists(sam_checkpoint):
            logger.error(f"SAM model checkpoint not found: {sam_checkpoint}")
            return False

        logger.info("Loading SAM model...")
        sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
        sam.to(device)
        predictor = SamPredictor(sam)
        sam_loaded = True
        logger.info("SAM model loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"Error loading SAM model: {str(e)}")
        return False

# ------------------------
# Database Initialization
# ------------------------
def init_db():
    """Initialize SQLite database"""
    try:
        conn = sqlite3.connect(app.config['DATABASE'])
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS potholes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                latitude REAL,
                longitude REAL,
                severity TEXT,
                area REAL,
                image_path TEXT,
                confidence REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'reported'
            )
        ''')
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully!")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")

# ------------------------
# Application Initialization
# ------------------------
def initialize_app():
    """Initialize SAM model and database before starting the server."""
    logger.info("Initializing application...")
    sam_ok = init_sam()
    init_db()
    
    if not sam_ok:
        logger.warning("SAM model not loaded. Detection will fail until SAM is available.")
    else:
        logger.info("SAM model is ready.")

# ------------------------
# Utility Functions
# ------------------------
def estimate_real_world_area(area_pixels, image_shape):
    """Estimate real-world area in square meters (simplified)"""
    pixels_per_meter = 100  # calibration required for real-world accuracy
    return area_pixels / (pixels_per_meter ** 2)

def determine_severity(area_m2):
    """Determine pothole severity based on area"""
    if area_m2 < 0.1:
        return 'low'
    elif area_m2 < 0.3:
        return 'medium'
    else:
        return 'high'

def create_overlay_image(image_np, mask):
    """Overlay detected pothole in red"""
    overlay = image_np.copy()
    overlay[mask > 0] = [255, 0, 0]
    return overlay

# ------------------------
# Routes
# ------------------------
@app.route('/')
def index():
    return render_template('index.html', sam_loaded=sam_loaded)

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'sam_loaded': sam_loaded,
        'database': app.config['DATABASE']
    })

@app.route('/detect', methods=['POST'])
def detect_pothole():
    if not sam_loaded:
        return jsonify({'error': 'SAM model not loaded'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    latitude = float(request.form.get('latitude', 0.0))
    longitude = float(request.form.get('longitude', 0.0))

    logger.info(f"Processing image from location: {latitude}, {longitude}")

    image = Image.open(image_file.stream).convert('RGB')
    image_np = np.array(image)

    # Set image for SAM predictor
    predictor.set_image(image_np)

    h, w = image_np.shape[:2]
    input_point = np.array([[w//2, h//2]])
    input_label = np.array([1])

    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False
    )

    if len(masks) == 0 or masks[0].size == 0:
        return jsonify({'error': 'No pothole detected'}), 400

    mask = masks[0]
    confidence = float(scores[0])
    area_pixels = np.sum(mask)
    area_m2 = estimate_real_world_area(area_pixels, image_np.shape)
    severity = determine_severity(area_m2)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"pothole_{timestamp}.jpg"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    overlay = create_overlay_image(image_np, mask)
    Image.fromarray(overlay).save(filepath)

    # Store in database
    conn = sqlite3.connect(app.config['DATABASE'])
    c = conn.cursor()
    c.execute('''
        INSERT INTO potholes (latitude, longitude, severity, area, image_path, confidence)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (latitude, longitude, severity, area_m2, filepath, confidence))
    pothole_id = c.lastrowid
    conn.commit()
    conn.close()

    # Broadcast to clients
    socketio.emit('new_pothole', {
        'id': pothole_id,
        'latitude': latitude,
        'longitude': longitude,
        'severity': severity,
        'area': area_m2,
        'confidence': confidence,
        'timestamp': datetime.now().isoformat()
    })

    logger.info(f"Pothole detected: ID {pothole_id}, Severity: {severity}, Area: {area_m2:.2f} m²")

    return jsonify({
        'success': True,
        'pothole_id': pothole_id,
        'severity': severity,
        'area_m2': area_m2,
        'confidence': confidence,
        'image_url': f'/image/{filename}'
    })

@app.route('/potholes')
def get_potholes():
    try:
        conn = sqlite3.connect(app.config['DATABASE'])
        c = conn.cursor()
        c.execute('SELECT * FROM potholes ORDER BY timestamp DESC')
        potholes = c.fetchall()
        conn.close()

        result = []
        for p in potholes:
            result.append({
                'id': p[0],
                'latitude': p[1],
                'longitude': p[2],
                'severity': p[3],
                'area': p[4],
                'image_path': p[5],
                'confidence': p[6],
                'timestamp': p[7],
                'status': p[8]
            })
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error getting potholes: {str(e)}")
        return jsonify([])

@app.route('/image/<filename>')
def get_image(filename):
    try:
        return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    except Exception as e:
        logger.error(f"Error serving image: {str(e)}")
        return jsonify({'error': 'Image not found'}), 404

@app.route('/map')
def show_map():
    try:
        conn = sqlite3.connect(app.config['DATABASE'])
        c = conn.cursor()
        c.execute('SELECT latitude, longitude, severity, id FROM potholes')
        potholes = c.fetchall()
        conn.close()

        center_lat, center_lon = (potholes[0][0], potholes[0][1]) if potholes else (40.7128, -74.0060)
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

        for lat, lon, severity, pid in potholes:
            color = 'red' if severity == 'high' else 'orange' if severity == 'medium' else 'green'
            folium.Marker(
                [lat, lon],
                popup=f'Pothole #{pid}<br>Severity: {severity}',
                icon=folium.Icon(color=color, icon='info-sign')
            ).add_to(m)

        return m._repr_html_()
    except Exception as e:
        logger.error(f"Error generating map: {str(e)}")
        return f"<h1>Error generating map: {str(e)}</h1>"

@app.route('/status')
def status():
    return jsonify({
        'sam_loaded': sam_loaded,
        'database_exists': os.path.exists(app.config['DATABASE']),
        'upload_folder_exists': os.path.exists(app.config['UPLOAD_FOLDER'])
    })

# ------------------------
# Main Entry Point
# ------------------------
if __name__ == '__main__':
    initialize_app()
    logger.info("Starting PotholeDetector server on http://0.0.0.0:5000 ...")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
