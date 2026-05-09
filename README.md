# 📱 Mobile Phone Detection System
### YOLOv8-Powered Interview Proctoring & Cheating Detection

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?logo=opencv)
![Flask](https://img.shields.io/badge/Flask-3.0%2B-lightgrey?logo=flask)
![CUDA](https://img.shields.io/badge/CUDA-GPU%20Accelerated-76b900?logo=nvidia)

</div>

---

## 🎯 Project Overview

A **real-time computer vision system** for detecting mobile phone usage during online interviews. Built with **YOLOv8** (COCO class 67 — "cell phone"), this system processes webcam feeds and video files, tracks detections across frames, reduces false positives, and generates structured alerts.

**Phase:** 3/4 — AI/Computer Vision | **Difficulty:** High

---

## 🏗️ Project Structure

```
Task 1/
│
├── 📁 models/              # YOLOv8 model weights (.pt files)
├── 📁 videos/              # Input test videos
├── 📁 outputs/             # Annotated output videos + CSV logs
├── 📁 logs/                # System logs
│
├── 📁 detection/
│   ├── __init__.py
│   └── detector.py         # YOLOv8 detector (COCO class 67)
│
├── 📁 tracking/
│   ├── __init__.py
│   └── tracker.py          # Persistence tracker + alert logic
│
├── 📁 api/
│   ├── __init__.py
│   └── routes.py           # Flask REST API endpoints
│
├── 📁 utils/
│   ├── __init__.py
│   ├── logger.py           # Structured logging
│   ├── helpers.py          # IoU, FPS, device detection
│   ├── visualization.py    # OpenCV drawing utilities
│   ├── metrics.py          # Precision/Recall/FPS metrics
│   └── export.py           # CSV/JSON export utilities
│
├── 📁 dashboard/
│   └── index.html          # Browser-based live detection UI
│
├── app.py                  # Main CLI entry point
├── inference.py            # End-to-end inference pipeline
├── config.py               # Centralized configuration
├── test_pipeline.py        # Automated validation tests
├── benchmark.py            # Performance benchmark suite
├── generate_test_video.py  # Synthetic test video generator
├── requirements.txt
├── Dockerfile              # Docker deployment
├── docker-compose.yml      # Docker Compose config
├── .dockerignore
├── .gitignore
└── README.md
```

---

## ⚡ Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

> **GPU (recommended):** Install PyTorch with CUDA first:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

### 2. Run Validation Tests (No Camera Needed)

```bash
python test_pipeline.py
```

### 3. Start API Server

```bash
python app.py
# API runs at: http://localhost:5000
```

### 4. Live Webcam Detection

```bash
python app.py --mode webcam
```

### 5. Process a Video File

```bash
python app.py --mode video --source videos/interview.mp4
```

### 6. Generate Test Video (No Real Video Needed)

```bash
python generate_test_video.py
```

### 7. Run Performance Benchmark

```bash
python benchmark.py --frames 100
```

---

## 🖥️ CLI Reference

```
python app.py [--mode {api,webcam,video}] [OPTIONS]

Modes:
  api       Start Flask REST API server (default)
  webcam    Live webcam detection
  video     Process video file

Options:
  --source PATH     Input video file (for video mode)
  --camera INT      Webcam device ID (default: 0)
  --host STR        API host (default: 0.0.0.0)
  --port INT        API port (default: 5000)
  --confidence FLT  Detection threshold (default: 0.40)
  --model PATH      YOLOv8 model path (default: yolov8n.pt)
  --device STR      cpu / cuda / mps / auto (default: auto)
  --save            Save annotated output video
  --no-display      Headless mode (no window)
  --output PATH     Output video path

Examples:
  python app.py --mode webcam --save
  python app.py --mode video --source clip.mp4 --confidence 0.5
  python app.py --mode api --port 8080 --debug
```

---

## 🌐 REST API

### Base URL: `http://localhost:5000`

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info & endpoint list |
| GET | `/status` | Health check + runtime stats |
| GET | `/model-info` | Model configuration |
| **POST** | **`/detect-phone`** | **Upload video/image for detection** |
| POST | `/detect-frame` | Base64 frame detection (webcam) |
| GET | `/history` | Detection history buffer |
| GET | `/alerts` | All suspicious timestamps |
| POST | `/reset-tracker` | Reset tracker state |
| GET | `/export-csv` | Export detection history to CSV |
| GET | `/export-alerts-csv` | Export alert timestamps to CSV |

---

### POST `/detect-phone`

Upload a video or image for phone detection.

**Request (multipart/form-data):**
```
file: <video or image file>
confidence: 0.4  (optional)
reset_tracker: false  (optional)
```

**Response:**
```json
{
  "phone_detected": true,
  "confidence": 0.91,
  "timestamp": "00:00:03.456",
  "frame_id": 103,
  "alert_active": true,
  "detections": [
    {
      "bbox": [145.2, 203.7, 312.8, 490.1],
      "confidence": 0.91,
      "avg_confidence": 0.88,
      "track_id": 1,
      "consecutive_frames": 7,
      "class_name": "cell phone"
    }
  ],
  "processing_time_ms": 47.3
}
```

---

### POST `/detect-frame`

For browser webcam integration (base64 encoded).

**Request (JSON):**
```json
{
  "frame": "<base64-encoded JPEG>",
  "frame_id": 42,
  "timestamp": 1.234,
  "confidence": 0.4
}
```

**Response:**
```json
{
  "phone_detected": true,
  "confidence": 0.87,
  "timestamp": "00:00:01.234",
  "frame_id": 42,
  "alert_active": false,
  "detections": [...],
  "processing_time_ms": 32.1
}
```

---

## 🤖 Model Explanation

### YOLOv8 Architecture

**YOLO** (You Only Look Once) is a single-stage real-time object detector.

- **Input:** Resized to 640×640 pixels
- **Backbone:** CSPDarknet (feature extraction)
- **Neck:** PANet (multi-scale feature fusion)  
- **Head:** Decoupled detection head per scale
- **Output:** Bounding boxes + confidence + class ID

### COCO Class 67 — "cell phone"

The pre-trained YOLOv8 model is trained on the **COCO dataset** (80 classes). We filter exclusively for class 67:

```python
results = model(frame, classes=[67])  # Only detect cell phones
```

### Model Variants

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| `yolov8n.pt` | 6 MB | ⚡⚡⚡ | ★★★ | Real-time webcam |
| `yolov8s.pt` | 22 MB | ⚡⚡ | ★★★★ | Balanced |
| `yolov8m.pt` | 52 MB | ⚡ | ★★★★★ | High accuracy |
| `yolov8l.pt` | 87 MB | 🐢 | ★★★★★ | Server/offline |

Default: **yolov8n** (nano) — auto-downloaded on first run.

---

## 🔁 Persistence Tracking Logic

The tracker prevents single-frame false positives by requiring a phone to appear for **N consecutive frames** before triggering an alert.

```
Frame 1: Phone detected → consecutive_frames = 1
Frame 2: Phone detected → consecutive_frames = 2
Frame 3: Phone detected → consecutive_frames = 3
Frame 4: Phone detected → consecutive_frames = 4
Frame 5: Phone detected → consecutive_frames = 5 → 🚨 ALERT TRIGGERED
Frame 6: Phone disappears → consecutive_frames resets to 0
```

### IoU-Based Matching

Each frame's detections are matched to existing tracks using **Intersection over Union (IoU)**:

```
IoU = Area of Overlap / Area of Union
```

- IoU ≥ 0.30 → same object (update track)
- IoU < 0.30 → new object (create track)

### Temporal Confidence Smoothing

Confidence is averaged over a sliding window of the last **10 frames**, preventing jitter from momentary high/low scores.

### Alert Cooldown

After an alert fires, a **5-second cooldown** prevents repeated notifications for the same event.

---

## 🛡️ False Positive Reduction

| Filter | Threshold | Purpose |
|--------|-----------|---------|
| Min bounding box area | 1,500 px² | Remove tiny distant detections |
| Max area ratio | 50% of frame | Remove full-screen objects |
| Aspect ratio | 0.2 – 3.0 | Remove overly thin/wide objects |
| Persistence threshold | 5 frames | Remove single-frame flashes |
| Confidence threshold | 0.40 | Remove low-certainty detections |
| Confidence averaging | 10-frame window | Smooth noisy scores |

---

## 📊 Performance Metrics

| Metric | CPU (yolov8n) | GPU (yolov8n) |
|--------|--------------|---------------|
| Inference speed | ~80ms/frame | ~12ms/frame |
| FPS | ~12 FPS | ~30+ FPS |
| Memory usage | ~500 MB RAM | ~1 GB VRAM |

> **Precision/Recall** depend on the specific dataset. With default settings:
> - Expected precision: ~85–92%
> - Expected recall: ~80–90%

Run `python benchmark.py` to measure on your hardware.

---

## ⚙️ Configuration

Edit `config.py` to tune system behavior:

```python
CONFIDENCE_THRESHOLD = 0.40    # Detection sensitivity
PERSISTENCE_THRESHOLD = 5      # Frames before alert
ALERT_COOLDOWN_SECONDS = 5.0   # Seconds between alerts
MAX_MISSING_FRAMES = 10        # Track expiry
MIN_BBOX_AREA = 1500           # False positive filter (px²)
```

---

## 🖼️ Visualization Features

- **Corner-accented bounding boxes** with color coding:
  - 🟢 Green — detection (below threshold)
  - 🟠 Orange — tracking (persistent)
  - 🔴 Red — alert active
- **Red alert banner** with confidence % at frame top
- **Live FPS counter** (color: green/yellow/red by speed)
- **Status panel** (bottom-right): phone detected, confidence, tracks, frame ID
- **Track ID + frame count** labels on each detection

---

## 🌐 Browser Dashboard

Open `dashboard/index.html` in a browser while the API is running to access the **Mobile Phone Detection** dashboard:

- **Live webcam monitoring** via browser
- **Real-time detection overlay** with bounding boxes
- **Confidence meter** and alert history
- **Configuration slider** for detection threshold
- **API connectivity indicator**

---

## 🐳 Docker Deployment

### Build and run:
```bash
docker-compose up --build
```

### Or without Docker Compose:
```bash
docker build -t phone-detector .
docker run -p 5000:5000 -v ./outputs:/app/outputs phone-detector
```

---

## 📋 Edge Case Handling

| Edge Case | Solution |
|-----------|----------|
| Partial phone visibility | Low confidence threshold (0.40) catches partial views |
| Blurry/low-res webcam | YOLOv8 handles variable input quality |
| Occluded objects | Temporal tracking fills brief occlusion gaps |
| TV remote / wallet | Aspect ratio + area filters exclude non-phone shapes |
| Multiple people | Per-object tracking with unique IDs |
| Fast hand movement | Exponential smoothing on bounding box coordinates |
| Poor lighting | YOLOv8 trained on diverse lighting conditions |

---

## 🔮 Future Enhancements

1. **Custom Training** — Fine-tune YOLOv8 on phone-specific datasets
2. **DeepSORT Integration** — Advanced multi-object tracking with ReID
3. **Multi-Camera Support** — Process multiple webcam streams simultaneously
4. **Sound Alert** — Audio notification on detection
5. **React Dashboard** — Real-time browser monitoring interface
6. **Gaze Detection** — Combine phone detection with eye-tracking
7. **FastAPI Migration** — Async API for higher throughput
8. **Export to ONNX/TensorRT** — Deployment optimization

---

## 📋 Requirements Summary

```
Python >= 3.9
ultralytics >= 8.0.0    # YOLOv8
opencv-python >= 4.8.0  # Computer vision
flask >= 3.0.0          # REST API
torch >= 2.0.0          # Deep learning
numpy >= 1.24.0         # Numerical computing
```

---

## 👨‍💻 Author

**UptoSkills Internship — Task 1**  
*AI/Computer Vision Track — Phase 3/4*

---

*Built with YOLOv8 + OpenCV + Flask | Production-Ready | GPU Accelerated*
