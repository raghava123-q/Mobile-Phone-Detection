"""
==============================================================
Configuration Module - Mobile Phone Detection System
==============================================================
Centralized configuration for all detection parameters,
model settings, tracking thresholds, and API configuration.
==============================================================
"""

import os
from pathlib import Path

# ==============================================================
# PATH CONFIGURATION
# ==============================================================
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
VIDEOS_DIR = BASE_DIR / "videos"
OUTPUTS_DIR = BASE_DIR / "outputs"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for directory in [MODELS_DIR, VIDEOS_DIR, OUTPUTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ==============================================================
# MODEL CONFIGURATION
# ==============================================================
# YOLOv8 model variant: 'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'
# 'n' = nano (fastest), 'x' = extra-large (most accurate)
YOLO_MODEL = "yolov8n.pt"

# COCO class ID for "cell phone" is 67
# Reference: https://docs.ultralytics.com/datasets/detect/coco/
PHONE_CLASS_ID = 67
PHONE_CLASS_NAME = "cell phone"

# ==============================================================
# DETECTION PARAMETERS
# ==============================================================
# Minimum confidence score to consider a detection valid
CONFIDENCE_THRESHOLD = 0.40

# IoU threshold for Non-Maximum Suppression
NMS_IOU_THRESHOLD = 0.45

# Input image size for YOLO inference (pixels)
INPUT_SIZE = 640

# Maximum number of detections per frame
MAX_DETECTIONS = 10

# ==============================================================
# PERSISTENCE TRACKING CONFIGURATION
# ==============================================================
# Number of consecutive frames a phone must appear before triggering alert
PERSISTENCE_THRESHOLD = 5

# Maximum frames a tracked object can be missing before removal
MAX_MISSING_FRAMES = 10

# IoU threshold for matching detections across frames
TRACKING_IOU_THRESHOLD = 0.3

# ==============================================================
# FALSE POSITIVE REDUCTION
# ==============================================================
# Minimum bounding box area (pixels²) to filter tiny detections
MIN_BBOX_AREA = 1500

# Maximum bounding box area ratio (relative to frame)
MAX_BBOX_AREA_RATIO = 0.5

# Aspect ratio constraints for phone-like objects
# Phones typically have aspect ratio between 0.3 and 0.8 (width/height)
MIN_ASPECT_RATIO = 0.2
MAX_ASPECT_RATIO = 3.0

# Confidence averaging window size
CONFIDENCE_WINDOW_SIZE = 10

# ==============================================================
# ALERT CONFIGURATION
# ==============================================================
# Cooldown period (seconds) between consecutive alerts
ALERT_COOLDOWN_SECONDS = 5.0

# Minimum average confidence to trigger alert
ALERT_MIN_AVG_CONFIDENCE = 0.45

# ==============================================================
# VISUALIZATION SETTINGS
# ==============================================================
# Bounding box colors (BGR format for OpenCV)
BBOX_COLOR_ALERT = (0, 0, 255)       # Red - phone detected
BBOX_COLOR_TRACKING = (0, 165, 255)  # Orange - tracking
BBOX_COLOR_NORMAL = (0, 255, 0)      # Green - no detection

# Alert banner settings
ALERT_BANNER_HEIGHT = 60
ALERT_BANNER_COLOR = (0, 0, 200)     # Dark red
ALERT_TEXT_COLOR = (255, 255, 255)    # White

# Font settings
FONT_SCALE = 0.6
FONT_THICKNESS = 2

# ==============================================================
# API CONFIGURATION
# ==============================================================
API_HOST = "0.0.0.0"
API_PORT = 5000
API_DEBUG = True

# Maximum upload file size (100 MB)
MAX_CONTENT_LENGTH = 100 * 1024 * 1024

# Allowed video extensions
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}

# ==============================================================
# PERFORMANCE SETTINGS
# ==============================================================
# Target FPS for webcam processing (0 = unlimited)
TARGET_FPS = 0

# Use GPU if available
USE_GPU = True

# Device selection: 'auto', 'cpu', 'cuda', 'mps'
DEVICE = "auto"

# ==============================================================
# LOGGING CONFIGURATION
# ==============================================================
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_FILE = LOGS_DIR / "detection.log"
