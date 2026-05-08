"""
==============================================================
Quick Test Script - Mobile Phone Detection System
==============================================================
Validates the complete pipeline without a real video/webcam.
Generates a synthetic test frame and runs detection + tracking.

Run:
    python test_pipeline.py
==============================================================
"""

import sys
import time
import json
import numpy as np
import cv2

print("=" * 60)
print("  Mobile Phone Detection System - Pipeline Validation")
print("=" * 60)
print()

# ── Step 1: Import check ──────────────────────────────────────
print("[1/5] Checking imports...")
try:
    from config import PHONE_CLASS_ID, CONFIDENCE_THRESHOLD, PERSISTENCE_THRESHOLD
    from utils.helpers import FPSCounter, get_device, calculate_iou, format_timestamp
    from utils.visualization import draw_alert_banner, draw_fps_counter, draw_status_panel
    from utils.logger import setup_logger
    from utils.metrics import DetectionMetrics
    from tracking.tracker import PersistenceTracker
    print("      [✓] All core modules imported successfully")
except ImportError as e:
    print(f"      [✗] Import error: {e}")
    sys.exit(1)

logger = setup_logger("Test")

# ── Step 2: Config validation ─────────────────────────────────
print("[2/5] Validating configuration...")
assert PHONE_CLASS_ID == 67, "Phone class ID should be 67 (COCO)"
assert 0 < CONFIDENCE_THRESHOLD < 1, "Confidence threshold must be between 0 and 1"
assert PERSISTENCE_THRESHOLD > 0, "Persistence threshold must be positive"
print(f"      [✓] PHONE_CLASS_ID     = {PHONE_CLASS_ID}")
print(f"      [✓] CONFIDENCE_THRESH  = {CONFIDENCE_THRESHOLD}")
print(f"      [✓] PERSISTENCE_THRESH = {PERSISTENCE_THRESHOLD}")

# ── Step 3: Tracker validation ────────────────────────────────
print("[3/5] Testing persistence tracker...")
tracker = PersistenceTracker()

# Simulate 8 consecutive frames with a phone detection
mock_detection = {
    "bbox": [100.0, 150.0, 300.0, 500.0],
    "confidence": 0.85,
    "class_id": 67,
    "class_name": "cell phone",
}

alert_triggered = False
for frame_i in range(8):
    tracked, alert = tracker.update([mock_detection], timestamp=frame_i * 0.033)
    if alert:
        alert_triggered = True
        break

assert alert_triggered, "Alert should trigger after PERSISTENCE_THRESHOLD frames"
print(f"      [✓] Alert triggered after {tracker.frame_count} frames")
print(f"      [✓] Tracker alert count = {tracker.alert_count}")

# Test false positive suppression (short flash)
tracker.reset()
for i in range(2):  # Only 2 frames — below threshold
    tracker.update([mock_detection], timestamp=i * 0.033)
tracker.update([], timestamp=0.1)  # Phone disappears
status = tracker.get_status()
assert not status["alert_active"], "No alert for transient detection"
print("      [✓] False positive (transient flash) correctly suppressed")

# ── Step 4: IoU calculation ───────────────────────────────────
print("[4/5] Testing IoU calculation...")
from utils.helpers import calculate_iou
box_a = np.array([10, 10, 100, 100])
box_b = np.array([50, 50, 150, 150])
iou = calculate_iou(box_a, box_b)
assert 0 < iou < 1, "IoU should be between 0 and 1 for partial overlap"
print(f"      [✓] IoU (50% overlap) = {iou:.4f}")

box_c = np.array([200, 200, 300, 300])
iou_zero = calculate_iou(box_a, box_c)
assert iou_zero == 0.0, "IoU should be 0 for non-overlapping boxes"
print(f"      [✓] IoU (no overlap)  = {iou_zero:.4f}")

# ── Step 5: Visualization test ────────────────────────────────
print("[5/5] Testing visualization pipeline...")
test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
test_frame[:] = (40, 40, 40)  # Dark grey background

# Draw something visible
cv2.rectangle(test_frame, (100, 100), (300, 400), (200, 200, 200), 2)
cv2.putText(test_frame, "Test Frame", (120, 250),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

# Apply visualization overlays
fps_frame = draw_fps_counter(test_frame.copy(), fps=28.5)
assert fps_frame.shape == test_frame.shape, "FPS overlay changed frame shape"

alert_frame = draw_alert_banner(test_frame.copy(), confidence=0.91)
assert alert_frame.shape == test_frame.shape, "Alert banner changed frame shape"

status_data = {
    "phone_detected": True,
    "confidence": 0.91,
    "tracked_objects": 1,
    "frame_id": 42,
    "alert_active": True,
}
status_frame = draw_status_panel(test_frame.copy(), status_data)
assert status_frame.shape == test_frame.shape, "Status panel changed frame shape"
print("      [✓] All visualization functions produce correct output shapes")

# Save test frame
from config import OUTPUTS_DIR
out_path = str(OUTPUTS_DIR / "test_frame_validation.jpg")
cv2.imwrite(out_path, alert_frame)
print(f"      [✓] Test frame saved: {out_path}")

# ── Step 6: Metrics validation ────────────────────────────────
print()
print("[+] Validating metrics module...")
metrics = DetectionMetrics(confidence_threshold=0.4)
for i in range(10):
    metrics.record_frame(
        {"phone_detected": True, "confidence": 0.82 + i * 0.01, "alert_active": i >= 4},
        ground_truth=True,
        inference_time_ms=22.5 + i * 0.5,
        fps=28.0 + i * 0.2,
    )
report = metrics.generate_report(save=False)
assert report["frame_statistics"]["total_frames"] == 10
assert report["performance_metrics"]["fps"]["avg_fps"] > 0
print("      [✓] Metrics accumulation and report generation working")

# ── Summary ───────────────────────────────────────────────────
print()
print("=" * 60)
print("  ALL TESTS PASSED ✓")
print("=" * 60)
print()
print("  Next steps:")
print("  1. Install dependencies:  pip install -r requirements.txt")
print("  2. Start API server:      python app.py")
print("  3. Test webcam:           python app.py --mode webcam")
print("  4. Process video:         python app.py --mode video --source video.mp4")
print()
