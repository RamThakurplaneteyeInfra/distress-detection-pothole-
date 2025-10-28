from flask import Flask, request, jsonify, render_template, send_file, abort
from flask_socketio import SocketIO
from flask_cors import CORS
from PIL import Image
import io, os, psycopg2, logging
from datetime import datetime
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
import folium
from fpdf import FPDF
from dotenv import load_dotenv
import threading
import time
import requests

# Load environment variables
load_dotenv()

# ------------------------
# Logging configuration
# ------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------
# Flask setup
# ------------------------
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'uploads')
app.config['DATABASE'] = os.getenv('DB_NAME', 'distress_db')
app.config['DB_HOST'] = os.getenv('DB_HOST', 'localhost')
app.config['DB_PORT'] = os.getenv('DB_PORT', '5432')
app.config['DB_USER'] = os.getenv('DB_USER', 'postgres')
app.config['DB_PASSWORD'] = os.getenv('DB_PASSWORD', 'admin')
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here')
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH', 16*1024*1024))
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ------------------------
# SAM Model Globals
# ------------------------
predictor = None
sam_loaded = False
sam_loading_status = "not_started"  # not_started, checking_model, downloading, loading, ready, failed

# ------------------------
# SAM Model Auto-Download Helper
# ------------------------
MODEL_FILENAME = "sam_vit_b_01ec64.pth"
MODEL_PATH = os.path.join(os.path.dirname(__file__), MODEL_FILENAME)
MODEL_URL = os.getenv("SAM_MODEL_URL", "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")

def ensure_sam_model():
    """Ensure the SAM model file is present; download if missing or incomplete."""
    try:
        # Quick size check — SAM base ~375 MB
        if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 350 * 1024 * 1024:
            logger.info(f"[Model Check] {MODEL_FILENAME} present ({os.path.getsize(MODEL_PATH)/(1024*1024):.1f} MB)")
            return True

        logger.warning(f"[Model Check] {MODEL_FILENAME} missing or incomplete — starting download...")
        with requests.get(MODEL_URL, stream=True, timeout=300) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        # Log every ~10 MB
                        if downloaded % (10 * 1024 * 1024) < 8192:
                            if total > 0:
                                percent = (downloaded / total) * 100
                                logger.info(f"[Model Download] {percent:.1f}% ({downloaded / (1024*1024):.1f} MB / {total / (1024*1024):.1f} MB)")
                            else:
                                logger.info(f"[Model Download] downloaded {downloaded / (1024*1024):.1f} MB")
        logger.info("[Model Check] SAM model downloaded successfully ✅")
        return True
    except Exception as e:
        logger.error(f"[Model Check] Failed to download SAM model: {e}")
        return False

# ------------------------
# SAM Initialization
# ------------------------
def init_sam():
    """Initialize the Segment Anything Model (SAM) and create a SamPredictor."""
    global predictor, sam_loaded, sam_loading_status
    try:
        sam_loading_status = "checking_model"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"[SAM Init] Using device: {device}")

        # Ensure model exists (download if necessary)
        sam_loading_status = "ensuring_model"
        if not ensure_sam_model():
            sam_loading_status = "failed"
            logger.error("[SAM Init] Cannot ensure SAM model file is present.")
            return False

        sam_loading_status = "loading"
        logger.info("[SAM Init] Loading SAM model into memory...")
        sam = sam_model_registry["vit_b"](checkpoint=MODEL_PATH)
        sam.to(device)

        predictor = SamPredictor(sam)

        # Warm up with a dummy image to allocate memory (do not run heavy ops here)
        try:
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            predictor.set_image(dummy)
            logger.info("[SAM Init] Warm-up set_image executed.")
        except Exception as e:
            # Not fatal — continue
            logger.warning(f"[SAM Init] Warm-up failed (non-fatal): {e}")

        sam_loaded = True
        sam_loading_status = "ready"
        logger.info("[SAM Init] SAM model loaded successfully ✅")
        return True
    except Exception as e:
        sam_loaded = False
        sam_loading_status = "failed"
        logger.error(f"[SAM Init] Error during initialization: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

# ------------------------
# Database helpers
# ------------------------
def get_db_connection():
    """Get PostgreSQL database connection. Supports DATABASE_URL env fallback."""
    try:
        database_url = os.getenv("DATABASE_URL")
        if database_url:
            # If a single DATABASE_URL is provided, use it (common on cloud providers)
            conn = psycopg2.connect(database_url)
        else:
            conn = psycopg2.connect(
                host=os.getenv('DB_HOST', app.config['DB_HOST']),
                port=os.getenv('DB_PORT', app.config['DB_PORT']),
                database=os.getenv('DB_NAME', app.config['DATABASE']),
                user=os.getenv('DB_USER', app.config['DB_USER']),
                password=os.getenv('DB_PASSWORD', app.config['DB_PASSWORD'])
            )
        return conn
    except psycopg2.Error as e:
        logger.error(f"[DB] Connection error: {e}")
        return None

def init_db():
    """Initialize PostgreSQL database and create tables"""
    conn = get_db_connection()
    if not conn:
        logger.warning("[DB] Failed to connect to database. App will continue without DB.")
        return False
    
    try:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS potholes (
                id SERIAL PRIMARY KEY,
                latitude DECIMAL(10, 8),
                longitude DECIMAL(11, 8),
                severity VARCHAR(20),
                area DECIMAL(10, 4),
                depth_meters DECIMAL(6, 3),
                image_path TEXT,
                confidence DECIMAL(6, 4),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status VARCHAR(20) DEFAULT 'reported'
            )
        ''')
        conn.commit()
        logger.info("[DB] Database initialized successfully")
        return True
    except psycopg2.Error as e:
        logger.warning(f"[DB] Initialization error: {e}. App will continue without DB.")
        return False
    finally:
        conn.close()

# ------------------------
# Utility
# ------------------------
def estimate_area(area_pixels):
    pixels_per_meter = 100  # adjust for real calibration
    return area_pixels / (pixels_per_meter**2)

def estimate_depth(area_m2):
    # Rough depth estimation: small area -> shallow, large area -> deeper
    return 0.05 + min(area_m2 * 0.5, 0.5)

def determine_severity(area_m2):
    if area_m2 < 0.1: return 'low'
    if area_m2 < 0.3: return 'medium'
    return 'high'

def overlay_image(image_np, mask):
    overlay = image_np.copy()
    overlay[mask > 0] = [255, 0, 0]
    return overlay

# ------------------------
# Routes
# ------------------------
@app.route('/')
def index():
    return render_template('index1.html', sam_loaded=sam_loaded, sam_loading_status=sam_loading_status)

@app.route('/health')
def health_check():
    """Provides a health check endpoint for the frontend to poll."""
    model_exists = os.path.exists(MODEL_PATH)
    model_size = os.path.getsize(MODEL_PATH) if model_exists else 0
    model_size_mb = model_size / (1024 * 1024) if model_exists else 0
    
    # Determine model status
    model_status = "not_found"
    if model_exists and model_size_mb > 350:
        model_status = "ready"
    elif model_exists and model_size_mb > 0:
        model_status = "downloading"
    elif model_exists:
        model_status = "invalid"
    
    return jsonify({
        'status': 'ok',
        'sam_loaded': sam_loaded,
        'sam_loading_status': sam_loading_status,
        'model_exists': model_exists,
        'model_size_mb': round(model_size_mb, 2) if model_exists else 0,
        'model_status': model_status,
        'sam_ready': sam_loaded and model_status == "ready",
        'checkpoint_path': MODEL_PATH
    })

@app.route('/debug')
def debug_info():
    """Detailed debug endpoint for troubleshooting."""
    model_exists = os.path.exists(MODEL_PATH)
    debug_info = {
        'sam_loaded': sam_loaded,
        'sam_loading_status': sam_loading_status,
        'model_file_exists': model_exists,
        'model_file_path': MODEL_PATH,
        'model_file_size_bytes': os.path.getsize(MODEL_PATH) if model_exists else 0,
        'model_file_size_mb': round(os.path.getsize(MODEL_PATH) / (1024 * 1024), 2) if model_exists else 0,
        'working_directory': os.getcwd(),
        'files_in_dir': sorted([f for f in os.listdir('.') if os.path.isfile(f)])[:50],
        'torch_device': "cuda" if torch.cuda.is_available() else "cpu"
    }
    return jsonify(debug_info)

@app.route('/detect', methods=['POST'])
def detect_pothole():
    if not sam_loaded or predictor is None:
        return jsonify({'error': 'SAM not loaded'}), 500
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    try:
        latitude = float(request.form.get('latitude', 0.0))
        longitude = float(request.form.get('longitude', 0.0))
    except Exception:
        latitude = 0.0
        longitude = 0.0

    image = Image.open(image_file.stream).convert('RGB')
    image_np = np.array(image)

    try:
        predictor.set_image(image_np)
        h, w = image_np.shape[:2]
        input_point = np.array([[w//2, h//2]])
        input_label = np.array([1])

        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False
        )
    except Exception as e:
        logger.error(f"[Detect] Predictor error: {e}")
        return jsonify({'error': 'Model inference error'}), 500

    if len(masks) == 0 or masks[0].size == 0:
        return jsonify({'success': False})

    mask = masks[0]
    confidence = float(scores[0]) if len(scores) > 0 else 0.0
    area_pixels = int(np.sum(mask))
    area_m2 = estimate_area(area_pixels)
    severity = determine_severity(area_m2)
    depth_meters = estimate_depth(area_m2)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"pothole_{timestamp}.jpg"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    overlay = overlay_image(image_np, mask)
    Image.fromarray(overlay).save(filepath)

    # Save to database if available
    pothole_id = None
    conn = get_db_connection()
    if conn:
        try:
            c = conn.cursor()
            c.execute('''
                INSERT INTO potholes (latitude, longitude, severity, area, depth_meters, image_path, confidence)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            ''', (latitude, longitude, severity, area_m2, depth_meters, filepath, confidence))
            row = c.fetchone()
            if row:
                pothole_id = int(row[0])
            conn.commit()
            socketio.emit('new_pothole', {
                'id': pothole_id,
                'latitude': latitude,
                'longitude': longitude,
                'severity': severity,
                'area': area_m2,
                'depth_meters': depth_meters,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            })
        except psycopg2.Error as e:
            logger.error(f"[DB] Insert error: {e}")
        finally:
            conn.close()
    else:
        logger.warning("[DB] No database connection available - results not saved to database")

    return jsonify({
        'success': True,
        'pothole_id': pothole_id,
        'severity': severity,
        'area_m2': area_m2,
        'depth_meters': depth_meters,
        'confidence': confidence,
        'image_url': f'/image/{filename}',
        'saved_to_db': pothole_id is not None
    })

@app.route('/potholes')
def get_potholes():
    conn = get_db_connection()
    if not conn:
        return jsonify([])  # Return empty list if no database connection
    
    try:
        c = conn.cursor()
        c.execute('SELECT * FROM potholes ORDER BY timestamp DESC')
        rows = c.fetchall()
        result = []
        for r in rows:
            result.append({
                'id': r[0],
                'latitude': float(r[1]) if r[1] else None,
                'longitude': float(r[2]) if r[2] else None,
                'severity': r[3],
                'area': float(r[4]) if r[4] else None,
                'depth_meters': float(r[5]) if r[5] else None,
                'image_path': r[6],
                'confidence': float(r[7]) if r[7] else None,
                'timestamp': r[8].isoformat() if r[8] else None,
                'status': r[9]
            })
        return jsonify(result)
    except psycopg2.Error as e:
        logger.error(f"[DB] Query error: {e}")
        return jsonify({'error': 'Failed to fetch potholes'}), 500
    finally:
        conn.close()

@app.route('/image/<filename>')
def get_image(filename):
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(path):
        return send_file(path)
    return abort(404)

@app.route('/export/<int:pothole_id>')
def export_pdf(pothole_id):
    conn = get_db_connection()
    if not conn:
        return "Database connection not available", 503
    
    try:
        c = conn.cursor()
        c.execute('SELECT * FROM potholes WHERE id=%s', (pothole_id,))
        row = c.fetchone()
        if not row: 
            return abort(404)
    except psycopg2.Error as e:
        logger.error(f"[DB] Query error: {e}")
        return abort(500)
    finally:
        conn.close()

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, f"Pothole Report #{row[0]}", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.ln(5)
    pdf.cell(0, 8, f"Latitude: {row[1]}", ln=True)
    pdf.cell(0, 8, f"Longitude: {row[2]}", ln=True)
    pdf.cell(0, 8, f"Severity: {row[3]}", ln=True)
    pdf.cell(0, 8, f"Area: {row[4]:.2f} m²", ln=True)
    pdf.cell(0, 8, f"Depth: {row[5]:.2f} m", ln=True)
    pdf.cell(0, 8, f"Confidence: {row[7]*100:.1f}%", ln=True)
    pdf.cell(0, 8, f"Timestamp: {row[8]}", ln=True)
    pdf.ln(5)
    if os.path.exists(row[6]):
        pdf.image(row[6], w=150)
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], f"pothole_report_{row[0]}.pdf")
    pdf.output(pdf_path)
    return send_file(pdf_path)

@app.route('/map')
def show_map():
    conn = get_db_connection()
    if not conn:
        center = (40.7128, -74.0060)  # Default center (New York)
        m = folium.Map(
            location=center, zoom_start=2,
            tiles='http://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
            attr='© Google'
        )
        return m._repr_html_()
    
    try:
        c = conn.cursor()
        c.execute('SELECT latitude, longitude, severity, id FROM potholes')
        rows = c.fetchall()
        center = (float(rows[0][0]), float(rows[0][1])) if rows else (40.7128, -74.0060)
    except psycopg2.Error as e:
        logger.error(f"[DB] Query error: {e}")
        return "Database query failed", 500
    finally:
        conn.close()
    m = folium.Map(
        location=center, zoom_start=13,
        tiles='http://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='© Google'
    )
    for lat, lon, severity, pid in rows:
        color = 'red' if severity=='high' else 'orange' if severity=='medium' else 'green'
        folium.Marker([float(lat), float(lon)], popup=f"Pothole #{pid}\nSeverity: {severity}", icon=folium.Icon(color=color)).add_to(m)
    return m._repr_html_()

# ------------------------
# Warm-up Hook
# ------------------------
@app.before_first_request
def warm_up_model():
    """Preload the SAM model so it's ready for the first request."""
    global predictor, sam_loaded
    if predictor is not None and sam_loaded:
        try:
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            predictor.set_image(dummy)
            logger.info("[Warm-Up] SAM model preloaded successfully.")
        except Exception as e:
            logger.warning(f"[Warm-Up] Failed (non-fatal): {e}")
    else:
        logger.info("[Warm-Up] Skipped - SAM still loading or predictor unavailable.")

# ------------------------
# Keep-Alive (optional)
# ------------------------
def keep_alive_loop():
    app_url = os.getenv("APP_URL")
    if not app_url:
        logger.info("[Keep-Alive] APP_URL not set; skipping keep-alive pings.")
        return
    while True:
        try:
            resp = requests.get(f"{app_url.rstrip('/')}/health", timeout=10)
            logger.info(f"[Keep-Alive] Pinged health: {resp.status_code}")
        except Exception as e:
            logger.warning(f"[Keep-Alive] Ping failed: {e}")
        time.sleep(900)  # 15 minutes

if os.getenv("KEEP_ALIVE", "false").lower() in ("1", "true", "yes"):
    t = threading.Thread(target=keep_alive_loop, daemon=True)
    t.start()
    logger.info("[Keep-Alive] Keep-alive thread started.")

# ------------------------
# App Initialization (DB + SAM loader)
# ------------------------
def initialize_app():
    # Initialize database first (fast)
    db_initialized = init_db()
    if db_initialized:
        logger.info("App initialized successfully with database")
    else:
        logger.info("App initialized successfully without database (database features may be limited)")

    # Initialize SAM model in background thread (slow)
    def load_sam_background():
        logger.info("Starting SAM model initialization in background...")
        success = init_sam()
        if success:
            logger.info("SAM model loaded successfully in background")
        else:
            logger.error("Failed to load SAM model in background")
    sam_thread = threading.Thread(target=load_sam_background, daemon=True)
    sam_thread.start()
    logger.info("SAM model loading started in background thread")

# Initialize the app when module is imported (works for both development and production)
initialize_app()

if __name__ == "__main__":
    port = int(os.getenv('PORT', '5000'))
    debug = os.getenv('FLASK_ENV') != 'production'
    socketio.run(app, host="0.0.0.0", port=port, debug=debug)
