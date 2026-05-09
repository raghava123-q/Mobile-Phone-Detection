"""
==============================================================
Flask REST API - Mobile Phone Detection System
==============================================================
Provides HTTP endpoints for:
  POST /detect-phone       - Detect phones in uploaded video/image
  POST /detect-frame       - Detect phone in a base64-encoded frame
  GET  /status             - API health check
  GET  /model-info         - Model configuration info
  GET  /history            - Detection history buffer
  GET  /alerts             - Suspicious timestamp log

CORS-enabled for browser-based frontend integration.
==============================================================
"""

import base64
import json
import tempfile
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

from config import (
    API_HOST,
    API_PORT,
    API_DEBUG,
    MAX_CONTENT_LENGTH,
    ALLOWED_EXTENSIONS,
    OUTPUTS_DIR,
    CONFIDENCE_THRESHOLD,
)
from detection.detector import PhoneDetector
from tracking.tracker import PersistenceTracker
from utils.helpers import FPSCounter, format_timestamp
from utils.logger import setup_logger

logger = setup_logger("FlaskAPI")

# ==============================================================
# Flask App Initialization
# ==============================================================
app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# ==============================================================
# Global detector and tracker (initialized once at startup)
# ==============================================================
detector: PhoneDetector = None
tracker: PersistenceTracker = None
fps_counter = FPSCounter(window_size=30)
api_start_time = time.time()
total_requests = 0


def get_detector() -> PhoneDetector:
    """Lazy-initialize the detector (loads model on first call)."""
    global detector
    if detector is None:
        logger.info("Initializing YOLOv8 detector...")
        detector = PhoneDetector()
        logger.info("[✓] Detector ready")
    return detector


def get_tracker() -> PersistenceTracker:
    """Lazy-initialize the tracker."""
    global tracker
    if tracker is None:
        tracker = PersistenceTracker()
        logger.info("[✓] Tracker ready")
    return tracker


# ==============================================================
# API ROUTES
# ==============================================================

@app.route("/", methods=["GET"])
def index():
    """Root endpoint — API welcome message."""
    return jsonify({
        "service": "Mobile Phone Detection API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "POST /detect-phone": "Upload video/image for phone detection",
            "POST /detect-frame": "Detect phone in base64-encoded frame",
            "GET  /status": "API health check",
            "GET  /model-info": "Model configuration",
            "GET  /history": "Detection history buffer",
            "GET  /alerts": "Suspicious timestamp log",
            "POST /reset-tracker": "Reset persistence tracker state",
        },
    })


@app.route("/status", methods=["GET"])
def status():
    """
    GET /status
    Health check and runtime statistics.
    """
    uptime = time.time() - api_start_time
    trk = get_tracker()
    trk_status = trk.get_status()

    return jsonify({
        "status": "healthy",
        "uptime_sec": round(uptime, 1),
        "total_requests": total_requests,
        "tracker": trk_status,
        "model_loaded": get_detector().model_loaded,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    })


@app.route("/model-info", methods=["GET"])
def model_info():
    """
    GET /model-info
    Return model configuration and performance info.
    """
    det = get_detector()
    return jsonify({
        "model_info": det.get_model_info(),
        "tracking_config": {
            "persistence_threshold": 5,
            "max_missing_frames": 10,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
        },
    })


@app.route("/detect-phone", methods=["POST"])
def detect_phone():
    """
    POST /detect-phone
    Detect mobile phones in an uploaded video or image file.

    Request:
        multipart/form-data:
            file: Video or image file
            confidence: float (optional, override threshold)
            reset_tracker: bool (optional, reset tracker state)

    Response:
        {
            "phone_detected": true,
            "confidence": 0.91,
            "timestamp": "00:00:01.234",
            "frame_id": 42,
            "alert_active": false,
            "detections": [...],
            "processing_time_ms": 45.2
        }
    """
    global total_requests
    total_requests += 1
    start = time.time()

    # Validate request
    if "file" not in request.files:
        return jsonify({"error": "No file provided. Use 'file' field in multipart form."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Get optional parameters
    confidence = request.form.get("confidence", CONFIDENCE_THRESHOLD, type=float)
    reset_tracker = request.form.get("reset_tracker", "false").lower() == "true"

    # Save uploaded file to temp location
    suffix = Path(file.filename).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        det = get_detector()
        trk = get_tracker()

        if reset_tracker:
            trk.reset()

        # Handle image files
        if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            frame = cv2.imread(tmp_path)
            if frame is None:
                return jsonify({"error": "Cannot decode image file"}), 400

            raw_detections = det.detect(frame, confidence_threshold=confidence)
            tracked_detections, alert_active = trk.update(raw_detections, timestamp=0.0)
            trk_status = trk.get_status()
            processing_ms = (time.time() - start) * 1000

            result = {
                "phone_detected": trk_status["phone_detected"],
                "confidence": round(trk_status["confidence"], 4),
                "timestamp": "00:00:00.000",
                "frame_id": 1,
                "alert_active": alert_active,
                "detections": _serialize_detections(tracked_detections),
                "processing_time_ms": round(processing_ms, 2),
                "source_type": "image",
                "filename": file.filename,
            }
            return jsonify(result)

        # Handle video files
        elif suffix in ALLOWED_EXTENSIONS:
            from inference import InferencePipeline
            pipeline = InferencePipeline(
                confidence_threshold=confidence,
                show_display=False,
                save_output=True,
            )
            summary = pipeline.process_video(tmp_path)
            processing_ms = (time.time() - start) * 1000
            summary["processing_time_ms"] = round(processing_ms, 2)
            summary["source_type"] = "video"
            summary["filename"] = file.filename
            return jsonify(summary)

        else:
            return jsonify({
                "error": f"Unsupported file type: {suffix}",
                "supported": list(ALLOWED_EXTENSIONS) + [".jpg", ".jpeg", ".png", ".bmp"]
            }), 415

    except Exception as e:
        logger.error(f"Detection error: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.route("/detect-frame", methods=["POST"])
def detect_frame():
    """
    POST /detect-frame
    Detect phone in a base64-encoded frame (for browser webcam integration).

    Request JSON:
        {
            "frame": "<base64-encoded image>",
            "frame_id": 42,
            "timestamp": 1.234,
            "confidence": 0.4
        }

    Response:
        {
            "phone_detected": true,
            "confidence": 0.91,
            "timestamp": "00:00:01.234",
            "frame_id": 42,
            "alert_active": false,
            "detections": [...]
        }
    """
    global total_requests
    total_requests += 1
    start = time.time()

    data = request.get_json()
    if not data or "frame" not in data:
        return jsonify({"error": "JSON body with 'frame' (base64) is required"}), 400

    # Decode base64 frame
    try:
        b64_data = data["frame"]
        # Remove data URL prefix if present (e.g., "data:image/jpeg;base64,...")
        if "," in b64_data:
            b64_data = b64_data.split(",", 1)[1]
        img_bytes = base64.b64decode(b64_data)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Cannot decode base64 frame"}), 400
    except Exception as e:
        return jsonify({"error": f"Frame decoding failed: {e}"}), 400

    frame_id = data.get("frame_id", 0)
    timestamp = data.get("timestamp", 0.0)
    confidence = data.get("confidence", CONFIDENCE_THRESHOLD)

    try:
        det = get_detector()
        trk = get_tracker()

        raw_detections = det.detect(frame, confidence_threshold=confidence)
        tracked_detections, alert_active = trk.update(raw_detections, timestamp=timestamp)
        trk_status = trk.get_status()
        processing_ms = (time.time() - start) * 1000

        return jsonify({
            "phone_detected": trk_status["phone_detected"],
            "confidence": round(trk_status["confidence"], 4),
            "timestamp": format_timestamp(timestamp),
            "frame_id": frame_id,
            "alert_active": alert_active,
            "detections": _serialize_detections(tracked_detections),
            "processing_time_ms": round(processing_ms, 2),
        })

    except Exception as e:
        logger.error(f"Frame detection error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/history", methods=["GET"])
def detection_history():
    """
    GET /history
    Return recent detection history buffer.

    Query params:
        limit: int — max records to return (default 50)
    """
    limit = request.args.get("limit", 50, type=int)
    trk = get_tracker()
    history = list(trk.detection_history)[-limit:]

    # Remove numpy arrays and annotated frames for JSON serialization
    clean_history = []
    for record in history:
        clean_record = {
            "frame_id": record["frame_id"],
            "timestamp": format_timestamp(record["timestamp"]),
            "alert_active": record["alert_active"],
            "detection_count": len(record["detections"]),
        }
        clean_history.append(clean_record)

    return jsonify({
        "count": len(clean_history),
        "history": clean_history,
    })


@app.route("/alerts", methods=["GET"])
def get_alerts():
    """
    GET /alerts
    Return all suspicious detection timestamps.
    """
    trk = get_tracker()
    return jsonify({
        "total_alerts": trk.alert_count,
        "alerts": trk.suspicious_timestamps,
    })


@app.route("/reset-tracker", methods=["POST"])
def reset_tracker():
    """
    POST /reset-tracker
    Reset the persistence tracker state (call between interview sessions).
    """
    trk = get_tracker()
    trk.reset()
    return jsonify({
        "message": "Tracker reset successfully",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    })


@app.route("/shutdown", methods=["POST"])
def shutdown():
    """
    POST /shutdown
    Gracefully stop the Flask API server.
    Used by the dashboard 'Stop Server' button.
    """
    import os
    import threading

    logger.info("Shutdown requested via API — stopping server...")

    def _shutdown():
        time.sleep(0.5)  # Let the response send first
        os._exit(0)

    threading.Thread(target=_shutdown, daemon=True).start()
    return jsonify({
        "message": "Server shutting down...",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    })


@app.route("/export-csv", methods=["GET"])
def export_csv():
    """
    GET /export-csv
    Export detection history to a downloadable CSV file.
    """
    from utils.export import export_detection_history_csv
    trk = get_tracker()
    history = list(trk.detection_history)

    if not history:
        return jsonify({"error": "No detection history to export"}), 404

    csv_path = export_detection_history_csv(history)
    return jsonify({
        "message": "CSV exported successfully",
        "file_path": csv_path,
        "records": len(history),
    })


@app.route("/export-alerts-csv", methods=["GET"])
def export_alerts_csv():
    """
    GET /export-alerts-csv
    Export alert timestamps to a downloadable CSV file.
    """
    from utils.export import export_alerts_csv as _export_alerts
    trk = get_tracker()

    if not trk.suspicious_timestamps:
        return jsonify({"error": "No alerts to export"}), 404

    csv_path = _export_alerts(trk.suspicious_timestamps)
    return jsonify({
        "message": "Alerts CSV exported successfully",
        "file_path": csv_path,
        "total_alerts": len(trk.suspicious_timestamps),
    })


# ==============================================================
# HELPER FUNCTIONS
# ==============================================================

def _serialize_detections(detections: list) -> list:
    """Serialize detection list for JSON response."""
    serialized = []
    for d in detections:
        serialized.append({
            "bbox": [round(v, 1) for v in d.get("bbox", [])],
            "confidence": round(d.get("confidence", 0), 4),
            "avg_confidence": round(d.get("avg_confidence", d.get("confidence", 0)), 4),
            "track_id": d.get("track_id"),
            "consecutive_frames": d.get("consecutive_frames", 0),
            "class_name": d.get("class_name", "cell phone"),
        })
    return serialized


@app.errorhandler(413)
def file_too_large(e):
    return jsonify({
        "error": "File too large",
        "max_size_mb": MAX_CONTENT_LENGTH // (1024 * 1024),
    }), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500
