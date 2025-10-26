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
# SAM Model
# ------------------------
predictor = None
sam_loaded = False

def init_sam():
    global predictor, sam_loaded
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        checkpoint_name = "sam_vit_b_01ec64.pth"
        checkpoint_path = os.path.join(os.path.dirname(__file__), checkpoint_name)

        # If model is not in the repo, download it as a fallback.
        if not os.path.exists(checkpoint_path):
            logger.warning(f"SAM checkpoint '{checkpoint_name}' not found. Downloading as a fallback...")
            model_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
            torch.hub.download_url_to_file(model_url, checkpoint_path)
            logger.info("SAM checkpoint downloaded successfully.")

        sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
        sam.to(device)
        predictor = SamPredictor(sam)
        sam_loaded = True
        logger.info("SAM loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"SAM init error: {str(e)}")
        return False

# ------------------------
# Database
# ------------------------
def get_db_connection():
    """Get PostgreSQL database connection"""
    try:
        conn = psycopg2.connect(
            host=app.config['DB_HOST'],
            port=app.config['DB_PORT'],
            database=app.config['DATABASE'],
            user=app.config['DB_USER'],
            password=app.config['DB_PASSWORD']
        )
        return conn
    except psycopg2.Error as e:
        logger.error(f"Database connection error: {e}")
        return None

def init_db():
    """Initialize PostgreSQL database and create tables"""
    conn = get_db_connection()
    if not conn:
        logger.error("Failed to connect to database")
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
                confidence DECIMAL(4, 3),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status VARCHAR(20) DEFAULT 'reported'
            )
        ''')
        conn.commit()
        logger.info("Database initialized successfully")
        return True
    except psycopg2.Error as e:
        logger.error(f"Database initialization error: {e}")
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
    # Example scaling: 0.05 m minimum, +0.2 m for large potholes
    return 0.05 + min(area_m2 * 0.5, 0.5)

def determine_severity(area_m2):
    if area_m2 < 0.1: return 'low'
    if area_m2 < 0.3: return 'medium'
    return 'high'

def overlay_image(image_np, mask):
    overlay = image_np.copy()
    overlay[mask>0] = [255,0,0]
    return overlay

# ------------------------
# Routes
# ------------------------
@app.route('/')
def index():
    return render_template('index1.html', sam_loaded=sam_loaded)

@app.route('/health')
def health_check():
    """Provides a health check endpoint for the frontend to poll."""
    return jsonify({
        'status': 'ok',
        'sam_loaded': sam_loaded
    })

@app.route('/detect', methods=['POST'])
def detect_pothole():
    if not sam_loaded:
        return jsonify({'error': 'SAM not loaded'}), 500
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    latitude = float(request.form.get('latitude', 0.0))
    longitude = float(request.form.get('longitude', 0.0))

    image = Image.open(image_file.stream).convert('RGB')
    image_np = np.array(image)

    predictor.set_image(image_np)
    h, w = image_np.shape[:2]
    
    # Try multiple points across the image for better detection
    # Sample points: center, left-center, right-center, top-center, bottom-center
    input_points = np.array([
        [w//2, h//2],        # Center
        [w//4, h//2],        # Left-center
        [3*w//4, h//2],      # Right-center
        [w//2, h//4],        # Top-center
        [w//2, 3*h//4]       # Bottom-center
    ])
    input_labels = np.array([1, 1, 1, 1, 1])

    # Try single point first (center)
    masks, scores, _ = predictor.predict(
        point_coords=input_points[[0]],
        point_labels=input_labels[[0]],
        multimask_output=True
    )

    if len(masks)==0 or masks[0].size==0:
        # Try with multiple points if single point fails
        masks, scores, _ = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True
        )
        
        if len(masks)==0 or masks[0].size==0:
            return jsonify({'success': False, 'error': 'No pothole detected'})

    mask = masks[0]
    confidence = float(scores[0])
    area_pixels = np.sum(mask)
    area_m2 = estimate_area(area_pixels)
    severity = determine_severity(area_m2)
    depth_meters = estimate_depth(area_m2)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"pothole_{timestamp}.jpg"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    overlay = overlay_image(image_np, mask)
    Image.fromarray(overlay).save(filepath)

    conn = get_db_connection()
    if not conn:
        return jsonify({'error': 'Database connection failed'}), 500
    
    try:
        c = conn.cursor()
        c.execute('''
            INSERT INTO potholes (latitude, longitude, severity, area, depth_meters, image_path, confidence)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        ''', (latitude, longitude, severity, area_m2, depth_meters, filepath, confidence))
        pothole_id = c.fetchone()[0]
        conn.commit()
    except psycopg2.Error as e:
        logger.error(f"Database insert error: {e}")
        return jsonify({'error': 'Failed to save pothole data'}), 500
    finally:
        conn.close()

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

    return jsonify({
        'success': True,
        'pothole_id': pothole_id,
        'severity': severity,
        'area_m2': area_m2,
        'depth_meters': depth_meters,
        'confidence': confidence,
        'image_url': f'/image/{filename}'
    })

@app.route('/potholes')
def get_potholes():
    conn = get_db_connection()
    if not conn:
        return jsonify({'error': 'Database connection failed'}), 500
    
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
        logger.error(f"Database query error: {e}")
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
        return abort(500)
    
    try:
        c = conn.cursor()
        c.execute('SELECT * FROM potholes WHERE id=%s', (pothole_id,))
        row = c.fetchone()
        if not row: 
            return abort(404)
    except psycopg2.Error as e:
        logger.error(f"Database query error: {e}")
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
        return "Database connection failed", 500
    
    try:
        c = conn.cursor()
        c.execute('SELECT latitude, longitude, severity, id FROM potholes')
        rows = c.fetchall()
        center = (float(rows[0][0]), float(rows[0][1])) if rows else (40.7128, -74.0060)
    except psycopg2.Error as e:
        logger.error(f"Database query error: {e}")
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
# Main
# ------------------------
def initialize_app():
    init_sam()
    if init_db():
        logger.info("App initialized successfully")
    else:
        logger.error("App initialization failed - database connection error")

if __name__ == "__main__":
    initialize_app()
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
