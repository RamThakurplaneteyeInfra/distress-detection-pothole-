from flask import Flask, request, jsonify, render_template, send_file, abort
from flask_socketio import SocketIO
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from PIL import Image
import io, os, logging, threading
from datetime import datetime, timezone
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
import folium
from fpdf import FPDF

# ------------------------
# Logging config
# ------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------
# Flask setup
# ------------------------
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Get the absolute path of the directory where app.py is located
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
# Use Render's persistent disk path if available, otherwise default to a local 'data' directory.
DATA_DIR = os.environ.get('RENDER_DISK_PATH', os.path.join(BASE_DIR, 'data'))
UPLOAD_FOLDER = os.path.join(DATA_DIR, 'uploads')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Use DATABASE_URL from environment if available, otherwise fall back to a local SQLite DB.
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', f'sqlite:///{os.path.join(DATA_DIR, "potholes.db")}')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'a-dev-secret')
app.config['MAX_CONTENT_LENGTH'] = 16*1024*1024  # 16 MB

# ------------------------
# SAM Model
# ------------------------
predictor = None
sam_loaded = False

def _load_sam_model_blocking():
    """The actual model loading logic. This is a blocking operation."""
    global predictor, sam_loaded
    if sam_loaded:
        return True
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        checkpoint_name = "sam_vit_b_01ec64.pth"
        checkpoint_path = os.path.join(os.path.dirname(__file__), checkpoint_name)

        # If model is not in the repo, download it as a fallback.
        if not os.path.exists(checkpoint_path):
            logger.warning(f"SAM checkpoint '{checkpoint_name}' not found. Downloading...")
            model_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
            torch.hub.download_url_to_file(model_url, checkpoint_path, progress=True)
            logger.info("SAM checkpoint downloaded successfully.")

        sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
        sam.to(device)
        predictor = SamPredictor(sam)
        sam_loaded = True
        logger.info("SAM model loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"SAM init error: {str(e)}")
        return False

def init_sam():
    """Starts the SAM model loading in a background thread to avoid blocking server startup."""
    if not sam_loaded:
        logger.info("Starting SAM model initialization in a background thread.")
        threading.Thread(target=_load_sam_model_blocking, daemon=True).start()

# ------------------------
# Database
# ------------------------
db = SQLAlchemy(app)

class Pothole(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    severity = db.Column(db.String(50))
    area = db.Column(db.Float)
    depth_meters = db.Column(db.Float)
    image_path = db.Column(db.String(255))
    confidence = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    status = db.Column(db.String(50), default='reported')

    def to_dict(self):
        return {
            'id': self.id,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'severity': self.severity,
            'area': self.area,
            'depth_meters': self.depth_meters,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'status': self.status
        }

def init_db():
    with app.app_context():
        db.create_all()
    logger.info("Database initialized")

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
    h,w = image_np.shape[:2]
    input_point = np.array([[w//2,h//2]])
    input_label = np.array([1])

    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False
    )

    if len(masks)==0 or masks[0].size==0:
        return jsonify({'success': False})

    mask = masks[0]
    confidence = float(scores[0])
    area_pixels = np.sum(mask)
    area_m2 = estimate_area(area_pixels)
    severity = determine_severity(area_m2)
    depth_meters = estimate_depth(area_m2)

    timestamp_utc = datetime.now(timezone.utc)
    filename = f"pothole_{timestamp_utc.strftime('%Y%m%d_%H%M%S')}.jpg"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    overlay = overlay_image(image_np, mask)
    Image.fromarray(overlay).save(filepath)

    new_pothole = Pothole(
        latitude=latitude, longitude=longitude, severity=severity,
        area=area_m2, depth_meters=depth_meters, image_path=filepath,
        confidence=confidence, timestamp=timestamp_utc
    )
    db.session.add(new_pothole)
    db.session.commit()

    pothole_id = new_pothole.id

    socketio.emit('new_pothole', {
        'id': new_pothole.id,
        'latitude': latitude,
        'longitude': longitude,
        'severity': severity,
        'area': area_m2,
        'depth_meters': depth_meters,
        'confidence': confidence,
        'timestamp': timestamp_utc.isoformat()
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
    potholes = Pothole.query.order_by(Pothole.timestamp.desc()).all()
    return jsonify([p.to_dict() for p in potholes])

@app.route('/image/<filename>')
def get_image(filename):
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(path):
        return send_file(path)
    return abort(404)

@app.route('/export/<int:pothole_id>')
def export_pdf(pothole_id):
    pothole = db.session.get(Pothole, pothole_id)
    if not pothole:
        return abort(404)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, f"Pothole Report #{pothole.id}", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.ln(5)
    pdf.cell(0, 8, f"Latitude: {pothole.latitude}", ln=True)
    pdf.cell(0, 8, f"Longitude: {pothole.longitude}", ln=True)
    pdf.cell(0, 8, f"Severity: {pothole.severity}", ln=True)
    pdf.cell(0, 8, f"Area: {pothole.area:.2f} m²", ln=True)
    pdf.cell(0, 8, f"Depth: {pothole.depth_meters:.2f} m", ln=True)
    pdf.cell(0, 8, f"Confidence: {pothole.confidence*100:.1f}%", ln=True)
    pdf.cell(0, 8, f"Timestamp: {pothole.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}", ln=True)
    pdf.ln(5)
    if pothole.image_path and os.path.exists(pothole.image_path):
        pdf.image(pothole.image_path, w=150)
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], f"pothole_report_{pothole.id}.pdf")
    pdf.output(pdf_path)
    return send_file(pdf_path)

@app.route('/map')
def show_map():
    potholes = Pothole.query.all()
    center = (potholes[0].latitude, potholes[0].longitude) if potholes else (40.7128, -74.0060)
    m = folium.Map(
        location=center, zoom_start=13,
        tiles='http://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='© Google'
    )
    for p in potholes:
        color = 'red' if p.severity=='high' else 'orange' if p.severity=='medium' else 'green'
        folium.Marker([p.latitude, p.longitude], popup=f"Pothole #{p.id}\nSeverity: {p.severity}", icon=folium.Icon(color=color)).add_to(m)
    return m._repr_html_()

# ------------------------
# Main
# ------------------------
def initialize_app():
    """Initializes database and starts background model loading."""
    # Ensure data directories exist before initializing anything.
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    init_sam()
    init_db()
    logger.info("App initialized")

def create_app():
    """App factory for Gunicorn to initialize before starting."""
    initialize_app()
    return app

if __name__ == "__main__":
    initialize_app()
    socketio.run(app, host="0.0.0.0", port=int(os.environ.get('PORT', 5000)), debug=True)
