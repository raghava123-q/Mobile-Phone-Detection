"""
==============================================================
Performance Benchmark - Mobile Phone Detection System
==============================================================
Measures: inference latency, FPS, model load time, tracker overhead.

Usage:
    python benchmark.py
    python benchmark.py --frames 200 --model yolov8s.pt
==============================================================
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from config import OUTPUTS_DIR, YOLO_MODEL
from utils.logger import setup_logger

logger = setup_logger("Benchmark")


def gen_frames(count=100, w=640, h=480):
    """Generate synthetic test frames."""
    for i in range(count):
        base = np.random.randint(30, 120, 3)
        frame = np.full((h, w, 3), base, dtype=np.uint8)
        noise = np.random.randint(-10, 11, frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        cx = w // 2 + np.random.randint(-50, 50)
        cv2.ellipse(frame, (cx, h // 3), (40, 50), 0, 0, 360, (120, 110, 100), -1)
        if np.random.random() < 0.3:
            px, py = np.random.randint(50, w - 100), np.random.randint(100, h - 150)
            pw = np.random.randint(40, 70)
            ph = int(pw * np.random.uniform(1.5, 2.2))
            cv2.rectangle(frame, (px, py), (px + pw, py + ph), (20, 20, 25), -1)
        yield frame


def run_benchmark(args):
    """Run complete benchmark suite."""
    print("\n" + "=" * 60)
    print("  Phone Detection — Performance Benchmark")
    print("=" * 60)

    results = {"timestamp": datetime.utcnow().isoformat() + "Z",
               "config": {"model": args.model, "num_frames": args.frames}}

    # 1. Model loading
    print("\n[1/4] Benchmarking model loading...")
    from detection.detector import PhoneDetector
    t0 = time.time()
    detector = PhoneDetector(model_path=args.model)
    load_time = time.time() - t0
    results["model_loading"] = {"load_time_sec": round(load_time, 3), "device": detector.device}
    print(f"      Load time: {load_time:.3f}s | Device: {detector.device}")

    # 2. Inference
    print(f"\n[2/4] Benchmarking inference ({args.frames} frames)...")
    latencies = []
    det_counts = []
    for i, frame in enumerate(gen_frames(args.frames)):
        t0 = time.time()
        dets = detector.detect(frame)
        latencies.append((time.time() - t0) * 1000)
        det_counts.append(len(dets))
        if (i + 1) % 20 == 0:
            avg = sum(latencies[-20:]) / 20
            print(f"      [{i+1}/{args.frames}] Avg: {avg:.1f}ms | FPS: {1000/avg:.1f}")

    arr = np.array(latencies)
    fps_arr = 1000 / arr
    results["inference"] = {
        "latency": {"mean_ms": round(float(np.mean(arr)), 2),
                     "p95_ms": round(float(np.percentile(arr, 95)), 2),
                     "p99_ms": round(float(np.percentile(arr, 99)), 2),
                     "min_ms": round(float(np.min(arr)), 2),
                     "max_ms": round(float(np.max(arr)), 2)},
        "fps": {"avg": round(float(np.mean(fps_arr)), 1),
                "min": round(float(np.min(fps_arr)), 1),
                "max": round(float(np.max(fps_arr)), 1)},
        "detections_total": int(sum(det_counts)),
    }

    # 3. Tracker overhead
    print(f"\n[3/4] Benchmarking tracker ({args.frames * 2} frames)...")
    from tracking.tracker import PersistenceTracker
    tracker = PersistenceTracker()
    mock = [{"bbox": [100, 150, 200, 350], "confidence": 0.85,
             "class_id": 67, "class_name": "cell phone"}]
    track_times = []
    for i in range(args.frames * 2):
        dets = mock if i % 3 != 0 else []
        t0 = time.time()
        tracker.update(dets, timestamp=i * 0.033)
        track_times.append((time.time() - t0) * 1e6)
    tarr = np.array(track_times)
    results["tracker"] = {"mean_us": round(float(np.mean(tarr)), 1),
                          "p95_us": round(float(np.percentile(tarr, 95)), 1),
                          "alerts": tracker.alert_count}
    print(f"      Mean overhead: {results['tracker']['mean_us']:.1f}µs | Alerts: {tracker.alert_count}")

    # 4. Visualization
    print("\n[4/4] Benchmarking visualization...")
    from utils.visualization import draw_detections, draw_alert_banner, draw_fps_counter, draw_status_panel
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (50, 50, 50)
    dets_v = [{"bbox": [100, 150, 250, 400], "confidence": 0.88,
               "class_name": "cell phone", "track_id": 1, "persistent_frames": 5}]
    status = {"phone_detected": True, "confidence": 0.88,
              "tracked_objects": 1, "frame_id": 100, "alert_active": True}
    viz_times = []
    for _ in range(50):
        t0 = time.time()
        out = draw_detections(frame, dets_v, is_alert=True)
        out = draw_alert_banner(out, confidence=0.88)
        out = draw_fps_counter(out, fps=30.0)
        out = draw_status_panel(out, status)
        viz_times.append((time.time() - t0) * 1000)
    varr = np.array(viz_times)
    results["visualization"] = {"mean_ms": round(float(np.mean(varr)), 2)}
    print(f"      Mean render: {results['visualization']['mean_ms']:.2f}ms")

    # Save report
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = OUTPUTS_DIR / f"benchmark_{ts}.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    inf = results["inference"]
    print("\n" + "=" * 60)
    print("  BENCHMARK COMPLETE")
    print("=" * 60)
    print(f"  Device:          {results['model_loading']['device']}")
    print(f"  Load time:       {results['model_loading']['load_time_sec']:.3f}s")
    print(f"  Avg inference:   {inf['latency']['mean_ms']:.1f}ms")
    print(f"  Avg FPS:         {inf['fps']['avg']:.1f}")
    print(f"  P95 latency:     {inf['latency']['p95_ms']:.1f}ms")
    print(f"  Tracker:         {results['tracker']['mean_us']:.1f}µs")
    print(f"  Render:          {results['visualization']['mean_ms']:.1f}ms")
    print(f"  Report:          {report_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phone Detection Benchmark")
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--model", type=str, default=YOLO_MODEL)
    run_benchmark(parser.parse_args())
