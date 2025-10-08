class PotholeDetector {
    constructor() {
        this.video = document.getElementById('video');
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.captureBtn = document.getElementById('captureBtn');
        this.uploadBtn = document.getElementById('uploadBtn');
        this.fileInput = document.getElementById('fileInput');
        this.locationBtn = document.getElementById('getLocation');
        this.detectionResult = document.getElementById('detectionResult');
        this.potholesList = document.getElementById('potholesList');
        this.pdfExportContainer = document.getElementById('pdfExportContainer');
        this.downloadPdfBtn = document.getElementById('downloadPdfBtn');
        
        this.currentLocation = { latitude: null, longitude: null };
        this.map = null;
        this.markers = [];
        this.socket = io();
        
        this.init();
    }

    async init() {
        await this.checkBackendStatus();
        await this.initCamera();
        this.initMap();
        this.initEventListeners();
        this.initSocket();
        this.loadPotholes();
    }

    async checkBackendStatus() {
        const statusEl = document.getElementById('statusAlert');
        const setStatus = (cls, text) => {
            statusEl.className = `alert ${cls} py-2`;
            statusEl.textContent = text;
            statusEl.style.display = 'block';
        };
        try {
            setStatus('alert-info', 'Checking backend status...');
            const res = await fetch('/health', { cache: 'no-store' });
            const json = await res.json();
            const ready = json && json.status === 'ok';
            const samReady = !!json.sam_loaded;
            this.captureBtn.disabled = !samReady;
            this.uploadBtn.disabled = !samReady;
            if (!samReady) {
                setStatus('alert-warning', 'Model is loading on server. Capture/Upload disabled until ready...');
                // Poll until model is ready
                const poll = async () => {
                    try {
                        const r = await fetch('/health', { cache: 'no-store' });
                        const j = await r.json();
                        if (j.sam_loaded) {
                            this.captureBtn.disabled = false;
                            this.uploadBtn.disabled = false;
                            setStatus('alert-success', 'Model ready. You can capture or upload now.');
                        } else {
                            setTimeout(poll, 5000);
                        }
                    } catch (e) {
                        setTimeout(poll, 5000);
                    }
                };
                setTimeout(poll, 5000);
            } else {
                setStatus('alert-success', 'Model ready. You can capture or upload now.');
            }
        } catch (e) {
            setStatus('alert-danger', `Backend check failed: ${e.message}`);
        }
    }

    async initCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
            this.video.srcObject = stream;
            this.video.addEventListener('loadedmetadata', () => {
                this.canvas.width = this.video.videoWidth;
                this.canvas.height = this.video.videoHeight;
            });
        } catch (err) {
            console.error('Error accessing camera:', err);
            this.detectionResult.innerHTML = `
                <div class="alert alert-warning">
                    Camera access denied or unavailable. On desktop, camera may be blocked; try image upload.
                </div>
            `;
        }
    }

    initMap() {
        this.map = L.map('map').setView([40.7128, -74.0060], 13);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors'
        }).addTo(this.map);
    }

    initEventListeners() {
        this.captureBtn.addEventListener('click', () => this.captureAndDetect());
        this.uploadBtn.addEventListener('click', () => this.fileInput.click());
        this.fileInput.addEventListener('change', (e) => this.handleFileUpload(e));
        this.locationBtn.addEventListener('click', () => this.getCurrentLocation());
    }

    initSocket() {
        this.socket.on('new_pothole', (pothole) => {
            this.addPotholeToList(pothole);
            this.addMarkerToMap(pothole);
            this.showNotification(`New pothole detected! Severity: ${pothole.severity}`);
        });
    }

    async captureAndDetect() {
        if (!this.currentLocation.latitude) {
            alert('Please get your location first');
            return;
        }
        this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
        this.canvas.toBlob(async (blob) => { await this.detectPothole(blob); }, 'image/jpeg');
    }

    async handleFileUpload(event) {
        const file = event.target.files[0];
        if (file) await this.detectPothole(file);
    }

    async detectPothole(imageBlob) {
        const formData = new FormData();
        formData.append('image', imageBlob);
        formData.append('latitude', this.currentLocation.latitude);
        formData.append('longitude', this.currentLocation.longitude);

        try {
            this.detectionResult.innerHTML = `
                <div class="alert alert-info">
                    <div class="spinner-border spinner-border-sm me-2"></div>
                    Detecting potholes...
                </div>
            `;

            const response = await fetch('/detect', { method: 'POST', body: formData });
            const result = await response.json();

            if (result.success) {
                const depth = result.depth_meters || 0.0;
                this.detectionResult.innerHTML = `
                    <div class="alert alert-success">
                        <h6>Pothole Detected!</h6>
                        <p><strong>Severity:</strong> ${result.severity}</p>
                        <p><strong>Area:</strong> ${result.area_m2.toFixed(2)} m²</p>
                        <p><strong>Estimated Depth:</strong> ${depth.toFixed(2)} m</p>
                        <p><strong>Confidence:</strong> ${(result.confidence*100).toFixed(1)}%</p>
                        <img src="${result.image_url}" class="img-fluid mt-2" alt="Detected pothole">
                    </div>
                `;
                // Enable PDF export
                this.pdfExportContainer.style.display = 'block';
                this.downloadPdfBtn.href = `/export/${result.pothole_id}`;
            } else {
                this.detectionResult.innerHTML = `
                    <div class="alert alert-warning">
                        No pothole detected. Please try a different image.
                    </div>
                `;
                this.pdfExportContainer.style.display = 'none';
            }
        } catch (error) {
            console.error('Detection error:', error);
            this.detectionResult.innerHTML = `
                <div class="alert alert-danger">
                    Error detecting pothole: ${error.message}
                </div>
            `;
        }
    }

    getCurrentLocation() {
        if (!navigator.geolocation) {
            alert('Geolocation not supported');
            return;
        }
        navigator.geolocation.getCurrentPosition(
            (position) => {
                this.currentLocation = {
                    latitude: position.coords.latitude,
                    longitude: position.coords.longitude
                };
                document.getElementById('latitude').textContent = this.currentLocation.latitude.toFixed(6);
                document.getElementById('longitude').textContent = this.currentLocation.longitude.toFixed(6);

                this.map.setView([this.currentLocation.latitude, this.currentLocation.longitude], 15);
                L.marker([this.currentLocation.latitude, this.currentLocation.longitude])
                    .addTo(this.map)
                    .bindPopup('Your current location')
                    .openPopup();
            },
            (error) => {
                const isSecure = location.protocol === 'https:';
                const hint = isSecure ? '' : '\nTip: Geolocation requires HTTPS to work in most browsers.';
                alert('Error getting location: ' + error.message + hint);
            },
            { enableHighAccuracy: true, timeout: 10000 }
        );
    }

    async loadPotholes() {
        try {
            const response = await fetch('/potholes');
            const potholes = await response.json();
            this.potholesList.innerHTML = '';
            this.clearMapMarkers();
            potholes.forEach(p => { this.addPotholeToList(p); this.addMarkerToMap(p); });
        } catch (error) { console.error('Error loading potholes:', error); }
    }

    addPotholeToList(pothole) {
        const severityClass = `severity-${pothole.severity}`;
        const date = new Date(pothole.timestamp).toLocaleString();
        const div = document.createElement('div');
        div.className = `card mb-2 ${severityClass}`;
        div.innerHTML = `
            <div class="card-body py-2">
                <h6 class="card-title mb-1">Pothole #${pothole.id}</h6>
                <p class="card-text mb-1">
                    <small>Severity: <span class="badge bg-${this.getSeverityColor(pothole.severity)}">${pothole.severity}</span></small>
                    <small>Area: ${pothole.area.toFixed(2)} m²</small>
                    <small>Depth: ${pothole.depth_meters ? pothole.depth_meters.toFixed(2) + ' m' : '-'}</small>
                </p>
                <p class="card-text mb-0"><small class="text-muted">${date}</small></p>
            </div>
        `;
        this.potholesList.prepend(div);
    }

    addMarkerToMap(pothole) {
        const marker = L.marker([pothole.latitude, pothole.longitude])
            .addTo(this.map)
            .bindPopup(`
                <strong>Pothole #${pothole.id}</strong><br>
                Severity: ${pothole.severity}<br>
                Area: ${pothole.area.toFixed(2)} m²<br>
                Depth: ${pothole.depth_meters ? pothole.depth_meters.toFixed(2)+' m' : '-'}<br>
                Confidence: ${(pothole.confidence*100).toFixed(1)}%
            `);
        this.markers.push(marker);
    }

    clearMapMarkers() {
        this.markers.forEach(marker => this.map.removeLayer(marker));
        this.markers = [];
    }

    getSeverityColor(severity) {
        switch (severity) {
            case 'high': return 'danger';
            case 'medium': return 'warning';
            case 'low': return 'success';
            default: return 'secondary';
        }
    }

    showNotification(message) {
        const notification = document.createElement('div');
        notification.className = 'alert alert-info alert-dismissible fade show position-fixed';
        notification.style.cssText = 'top:20px;right:20px;z-index:1000;min-width:300px;';
        notification.innerHTML = `${message}<button type="button" class="btn-close" data-bs-dismiss="alert"></button>`;
        document.body.appendChild(notification);
        setTimeout(() => { if (notification.parentNode) notification.parentNode.removeChild(notification); }, 5000);
    }
}

document.addEventListener('DOMContentLoaded', () => { new PotholeDetector(); });
class PotholeDetector {
    constructor() {
        this.video = document.getElementById('video');
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.captureBtn = document.getElementById('captureBtn');
        this.uploadBtn = document.getElementById('uploadBtn');
        this.fileInput = document.getElementById('fileInput');
        this.locationBtn = document.getElementById('getLocation');
        this.detectionResult = document.getElementById('detectionResult');
        this.potholesList = document.getElementById('potholesList');
        this.pdfExportContainer = document.getElementById('pdfExportContainer');
        this.downloadPdfBtn = document.getElementById('downloadPdfBtn');
        
        this.currentLocation = { latitude: null, longitude: null };
        this.map = null;
        this.markers = [];
        this.socket = io();
        
        this.init();
    }

    async init() {
        await this.initCamera();
        this.initMap();
        this.initEventListeners();
        this.initSocket();
        this.loadPotholes();
    }

    async initCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
            this.video.srcObject = stream;
            this.video.addEventListener('loadedmetadata', () => {
                this.canvas.width = this.video.videoWidth;
                this.canvas.height = this.video.videoHeight;
            });
        } catch (err) {
            console.error('Error accessing camera:', err);
            this.detectionResult.innerHTML = `
                <div class="alert alert-warning">
                    Camera access denied. Please use image upload instead.
                </div>
            `;
        }
    }

    initMap() {
        this.map = L.map('map').setView([40.7128, -74.0060], 13);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors'
        }).addTo(this.map);
    }

    initEventListeners() {
        this.captureBtn.addEventListener('click', () => this.captureAndDetect());
        this.uploadBtn.addEventListener('click', () => this.fileInput.click());
        this.fileInput.addEventListener('change', (e) => this.handleFileUpload(e));
        this.locationBtn.addEventListener('click', () => this.getCurrentLocation());
    }

    initSocket() {
        this.socket.on('new_pothole', (pothole) => {
            this.addPotholeToList(pothole);
            this.addMarkerToMap(pothole);
            this.showNotification(`New pothole detected! Severity: ${pothole.severity}`);
        });
    }

    async captureAndDetect() {
        if (!this.currentLocation.latitude) {
            alert('Please get your location first');
            return;
        }
        this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
        this.canvas.toBlob(async (blob) => { await this.detectPothole(blob); }, 'image/jpeg');
    }

    async handleFileUpload(event) {
        const file = event.target.files[0];
        if (file) await this.detectPothole(file);
    }

    async detectPothole(imageBlob) {
        const formData = new FormData();
        formData.append('image', imageBlob);
        formData.append('latitude', this.currentLocation.latitude);
        formData.append('longitude', this.currentLocation.longitude);

        try {
            this.detectionResult.innerHTML = `
                <div class="alert alert-info">
                    <div class="spinner-border spinner-border-sm me-2"></div>
                    Detecting potholes...
                </div>
            `;

            const response = await fetch('/detect', { method: 'POST', body: formData });
            const result = await response.json();

            if (result.success) {
                const depth = result.depth_meters || 0.0;
                this.detectionResult.innerHTML = `
                    <div class="alert alert-success">
                        <h6>Pothole Detected!</h6>
                        <p><strong>Severity:</strong> ${result.severity}</p>
                        <p><strong>Area:</strong> ${result.area_m2.toFixed(2)} m²</p>
                        <p><strong>Estimated Depth:</strong> ${depth.toFixed(2)} m</p>
                        <p><strong>Confidence:</strong> ${(result.confidence*100).toFixed(1)}%</p>
                        <img src="${result.image_url}" class="img-fluid mt-2" alt="Detected pothole">
                    </div>
                `;
                // Enable PDF export
                this.pdfExportContainer.style.display = 'block';
                this.downloadPdfBtn.href = `/export/${result.pothole_id}`;
            } else {
                this.detectionResult.innerHTML = `
                    <div class="alert alert-warning">
                        No pothole detected. Please try a different image.
                    </div>
                `;
                this.pdfExportContainer.style.display = 'none';
            }
        } catch (error) {
            console.error('Detection error:', error);
            this.detectionResult.innerHTML = `
                <div class="alert alert-danger">
                    Error detecting pothole: ${error.message}
                </div>
            `;
        }
    }

    getCurrentLocation() {
        if (!navigator.geolocation) {
            alert('Geolocation not supported');
            return;
        }
        navigator.geolocation.getCurrentPosition(
            (position) => {
                this.currentLocation = {
                    latitude: position.coords.latitude,
                    longitude: position.coords.longitude
                };
                document.getElementById('latitude').textContent = this.currentLocation.latitude.toFixed(6);
                document.getElementById('longitude').textContent = this.currentLocation.longitude.toFixed(6);

                this.map.setView([this.currentLocation.latitude, this.currentLocation.longitude], 15);
                L.marker([this.currentLocation.latitude, this.currentLocation.longitude])
                    .addTo(this.map)
                    .bindPopup('Your current location')
                    .openPopup();
            },
            (error) => { alert('Error getting location: ' + error.message); }
        );
    }

    async loadPotholes() {
        try {
            const response = await fetch('/potholes');
            const potholes = await response.json();
            this.potholesList.innerHTML = '';
            this.clearMapMarkers();
            potholes.forEach(p => { this.addPotholeToList(p); this.addMarkerToMap(p); });
        } catch (error) { console.error('Error loading potholes:', error); }
    }

    addPotholeToList(pothole) {
        const severityClass = `severity-${pothole.severity}`;
        const date = new Date(pothole.timestamp).toLocaleString();
        const div = document.createElement('div');
        div.className = `card mb-2 ${severityClass}`;
        div.innerHTML = `
            <div class="card-body py-2">
                <h6 class="card-title mb-1">Pothole #${pothole.id}</h6>
                <p class="card-text mb-1">
                    <small>Severity: <span class="badge bg-${this.getSeverityColor(pothole.severity)}">${pothole.severity}</span></small>
                    <small>Area: ${pothole.area.toFixed(2)} m²</small>
                    <small>Depth: ${pothole.depth_meters ? pothole.depth_meters.toFixed(2) + ' m' : '-'}</small>
                </p>
                <p class="card-text mb-0"><small class="text-muted">${date}</small></p>
            </div>
        `;
        this.potholesList.prepend(div);
    }

    addMarkerToMap(pothole) {
        const marker = L.marker([pothole.latitude, pothole.longitude])
            .addTo(this.map)
            .bindPopup(`
                <strong>Pothole #${pothole.id}</strong><br>
                Severity: ${pothole.severity}<br>
                Area: ${pothole.area.toFixed(2)} m²<br>
                Depth: ${pothole.depth_meters ? pothole.depth_meters.toFixed(2)+' m' : '-'}<br>
                Confidence: ${(pothole.confidence*100).toFixed(1)}%
            `);
        this.markers.push(marker);
    }

    clearMapMarkers() {
        this.markers.forEach(marker => this.map.removeLayer(marker));
        this.markers = [];
    }

    getSeverityColor(severity) {
        switch (severity) {
            case 'high': return 'danger';
            case 'medium': return 'warning';
            case 'low': return 'success';
            default: return 'secondary';
        }
    }

    showNotification(message) {
        const notification = document.createElement('div');
        notification.className = 'alert alert-info alert-dismissible fade show position-fixed';
        notification.style.cssText = 'top:20px;right:20px;z-index:1000;min-width:300px;';
        notification.innerHTML = `${message}<button type="button" class="btn-close" data-bs-dismiss="alert"></button>`;
        document.body.appendChild(notification);
        setTimeout(() => { if (notification.parentNode) notification.parentNode.removeChild(notification); }, 5000);
    }
}

document.addEventListener('DOMContentLoaded', () => { new PotholeDetector(); });
