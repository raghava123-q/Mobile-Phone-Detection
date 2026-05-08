"""
==============================================================
Performance Metrics - Mobile Phone Detection System
==============================================================
Computes and reports detection performance metrics:
  - Precision, Recall, F1 Score
  - Average FPS
  - Inference latency (mean / p95 / p99)
  - Confusion matrix data
  - Per-class statistics

Used for benchmarking and system validation.
==============================================================
"""

import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import OUTPUTS_DIR
from utils.logger import setup_logger

logger = setup_logger("Metrics")


class DetectionMetrics:
    """
    Computes and tracks detection performance metrics.

    Supports:
        - Real-time metric accumulation during processing
        - Precision / Recall / F1 calculation
        - Latency percentile analysis
        - JSON and console report generation
    """

    def __init__(self, confidence_threshold: float = 0.4):
        """
        Initialize metrics tracker.

        Args:
            confidence_threshold: Threshold used during evaluation
        """
        self.confidence_threshold = confidence_threshold
        self.reset()

    def reset(self):
        """Reset all accumulated metrics."""
        self.tp = 0          # True positives
        self.fp = 0          # False positives
        self.fn = 0          # False negatives
        self.tn = 0          # True negatives

        self.total_frames = 0
        self.phone_detected_frames = 0
        self.alert_frames = 0

        self.confidences = []       # All detection confidence scores
        self.inference_times = []   # Per-frame inference times (ms)
        self.fps_values = []        # Per-frame FPS measurements

        self.start_time = time.time()

    def record_frame(
        self,
        detection_result: Dict,
        ground_truth: bool = None,
        inference_time_ms: float = None,
        fps: float = None,
    ):
        """
        Record metrics for a single frame.

        Args:
            detection_result: Result dict from InferencePipeline.process_frame()
            ground_truth: True if phone is actually present (for supervised eval)
            inference_time_ms: Frame inference time in milliseconds
            fps: Current FPS
        """
        self.total_frames += 1
        predicted = detection_result.get("phone_detected", False)
        conf = detection_result.get("confidence", 0.0)

        if predicted:
            self.phone_detected_frames += 1
            if conf > 0:
                self.confidences.append(conf)

        if detection_result.get("alert_active", False):
            self.alert_frames += 1

        if inference_time_ms is not None:
            self.inference_times.append(inference_time_ms)

        if fps is not None and fps > 0:
            self.fps_values.append(fps)

        # Update confusion matrix if ground truth is provided
        if ground_truth is not None:
            if predicted and ground_truth:
                self.tp += 1
            elif predicted and not ground_truth:
                self.fp += 1
            elif not predicted and ground_truth:
                self.fn += 1
            else:
                self.tn += 1

    def compute_precision(self) -> float:
        """Precision = TP / (TP + FP)"""
        if (self.tp + self.fp) == 0:
            return 0.0
        return self.tp / (self.tp + self.fp)

    def compute_recall(self) -> float:
        """Recall = TP / (TP + FN)"""
        if (self.tp + self.fn) == 0:
            return 0.0
        return self.tp / (self.tp + self.fn)

    def compute_f1(self) -> float:
        """F1 Score = 2 * (Precision * Recall) / (Precision + Recall)"""
        precision = self.compute_precision()
        recall = self.compute_recall()
        if (precision + recall) == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def compute_accuracy(self) -> float:
        """Accuracy = (TP + TN) / Total"""
        total = self.tp + self.fp + self.fn + self.tn
        if total == 0:
            return 0.0
        return (self.tp + self.tn) / total

    def get_latency_stats(self) -> Dict:
        """
        Compute inference latency statistics.

        Returns:
            Dict with mean, std, min, max, p95, p99 in milliseconds
        """
        if not self.inference_times:
            return {"mean": 0, "std": 0, "min": 0, "max": 0, "p95": 0, "p99": 0}

        arr = np.array(self.inference_times)
        return {
            "mean_ms": round(float(np.mean(arr)), 2),
            "std_ms": round(float(np.std(arr)), 2),
            "min_ms": round(float(np.min(arr)), 2),
            "max_ms": round(float(np.max(arr)), 2),
            "p95_ms": round(float(np.percentile(arr, 95)), 2),
            "p99_ms": round(float(np.percentile(arr, 99)), 2),
        }

    def get_fps_stats(self) -> Dict:
        """Compute FPS statistics."""
        if not self.fps_values:
            elapsed = time.time() - self.start_time
            avg = self.total_frames / elapsed if elapsed > 0 else 0
            return {"avg_fps": round(avg, 1), "min_fps": 0, "max_fps": 0}

        arr = np.array(self.fps_values)
        return {
            "avg_fps": round(float(np.mean(arr)), 1),
            "min_fps": round(float(np.min(arr)), 1),
            "max_fps": round(float(np.max(arr)), 1),
        }

    def get_confidence_stats(self) -> Dict:
        """Compute detection confidence statistics."""
        if not self.confidences:
            return {"mean": 0, "std": 0, "min": 0, "max": 0}

        arr = np.array(self.confidences)
        return {
            "mean": round(float(np.mean(arr)), 4),
            "std": round(float(np.std(arr)), 4),
            "min": round(float(np.min(arr)), 4),
            "max": round(float(np.max(arr)), 4),
        }

    def generate_report(self, save: bool = True) -> Dict:
        """
        Generate a comprehensive performance report.

        Args:
            save: Whether to save the report as JSON

        Returns:
            Complete metrics report dictionary
        """
        elapsed = time.time() - self.start_time
        detection_rate = (
            self.phone_detected_frames / self.total_frames
            if self.total_frames > 0 else 0
        )

        report = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "evaluation_duration_sec": round(elapsed, 2),
            "confidence_threshold": self.confidence_threshold,

            "frame_statistics": {
                "total_frames": self.total_frames,
                "phone_detected_frames": self.phone_detected_frames,
                "alert_frames": self.alert_frames,
                "detection_rate": round(detection_rate, 4),
            },

            "classification_metrics": {
                "true_positives": self.tp,
                "false_positives": self.fp,
                "false_negatives": self.fn,
                "true_negatives": self.tn,
                "precision": round(self.compute_precision(), 4),
                "recall": round(self.compute_recall(), 4),
                "f1_score": round(self.compute_f1(), 4),
                "accuracy": round(self.compute_accuracy(), 4),
                "note": "Requires ground truth labels. 0 if not provided.",
            },

            "performance_metrics": {
                "fps": self.get_fps_stats(),
                "latency": self.get_latency_stats(),
                "confidence": self.get_confidence_stats(),
            },
        }

        if save:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = OUTPUTS_DIR / f"metrics_report_{ts}.json"
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Metrics report saved: {report_path}")

        return report

    def print_summary(self):
        """Print a formatted metrics summary to console."""
        report = self.generate_report(save=False)
        fps = report["performance_metrics"]["fps"]
        latency = report["performance_metrics"]["latency"]
        metrics = report["classification_metrics"]
        frame_stats = report["frame_statistics"]

        print()
        print("=" * 55)
        print("  PERFORMANCE METRICS REPORT")
        print("=" * 55)
        print(f"  Total Frames:        {frame_stats['total_frames']}")
        print(f"  Detections:          {frame_stats['phone_detected_frames']} frames ({frame_stats['detection_rate']:.1%})")
        print(f"  Alert Frames:        {frame_stats['alert_frames']}")
        print()
        print(f"  Average FPS:         {fps['avg_fps']}")
        print(f"  Min / Max FPS:       {fps['min_fps']} / {fps['max_fps']}")
        print()
        if latency["mean_ms"] > 0:
            print(f"  Avg Latency:         {latency['mean_ms']} ms")
            print(f"  P95 Latency:         {latency['p95_ms']} ms")
            print(f"  P99 Latency:         {latency['p99_ms']} ms")
            print()
        if self.tp + self.fp + self.fn + self.tn > 0:
            print(f"  Precision:           {metrics['precision']:.4f}")
            print(f"  Recall:              {metrics['recall']:.4f}")
            print(f"  F1 Score:            {metrics['f1_score']:.4f}")
            print(f"  Accuracy:            {metrics['accuracy']:.4f}")
        print("=" * 55)
        print()
