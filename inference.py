"""
==============================================================
Inference Engine - Mobile Phone Detection System
==============================================================
Main inference pipeline that combines the YOLOv8 detector
with the persistence tracker to process:
  - Webcam live streams
  - Video files
  - Single images

Produces:
  - Annotated output video files
  - Real-time JSON detection results
  - Suspicious activity log (CSV)
  - Detection statistics
==============================================================
"""

import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np

from config import (
    OUTPUTS_DIR,
    CONFIDENCE_THRESHOLD,
    PERSISTENCE_THRESHOLD,
)
from detection.detector import PhoneDetector
from tracking.tracker import PersistenceTracker
from utils.helpers import FPSCounter, format_timestamp, validate_video_file
from utils.logger import setup_logger
from utils.visualization import (
    draw_detections,
    draw_alert_banner,
    draw_fps_counter,
    draw_status_panel,
)

logger = setup_logger("Inference")


class InferencePipeline:
    """
    End-to-end inference pipeline for phone detection.

    Combines YOLOv8 detection, persistence tracking, and
    visualization into a single easy-to-use interface.

    Usage:
        pipeline = InferencePipeline()

        # Process a video file
        results = pipeline.process_video("interview.mp4")

        # Process webcam stream
        pipeline.process_webcam()

        # Process a single frame
        result = pipeline.process_frame(frame, frame_id=0, timestamp=0.0)
    """

    def __init__(
        self,
        model_path: str = None,
        confidence_threshold: float = None,
        device: str = "auto",
        show_display: bool = True,
        save_output: bool = True,
    ):
        """
        Initialize the inference pipeline.

        Args:
            model_path: YOLOv8 model weights path
            confidence_threshold: Minimum detection confidence
            device: Compute device
            show_display: Whether to display live frames in a window
            save_output: Whether to save annotated video output
        """
        self.show_display = show_display
        self.save_output = save_output

        logger.info("=" * 60)
        logger.info("  Mobile Phone Detection System - Inference Pipeline")
        logger.info("=" * 60)

        # Initialize detector and tracker
        self.detector = PhoneDetector(
            model_path=model_path,
            device=device,
            confidence_threshold=confidence_threshold or CONFIDENCE_THRESHOLD,
        )
        self.tracker = PersistenceTracker()
        self.fps_counter = FPSCounter(window_size=30)

        logger.info("[✓] Inference pipeline ready")

    def process_frame(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
        timestamp: float = 0.0,
    ) -> Dict:
        """
        Process a single frame through the full pipeline.

        Pipeline steps:
            1. Run YOLOv8 detection
            2. Update persistence tracker
            3. Build annotated frame
            4. Return structured JSON result

        Args:
            frame: Input frame (BGR numpy array)
            frame_id: Frame index number
            timestamp: Frame timestamp in seconds

        Returns:
            Detection result dictionary:
            {
                "phone_detected": bool,
                "confidence": float,
                "timestamp": "00:00:01.234",
                "frame_id": 42,
                "alert_active": bool,
                "detections": [...],
                "annotated_frame": np.ndarray
            }
        """
        # Step 1: Detect phones in frame
        raw_detections = self.detector.detect(frame)

        # Step 2: Update tracker with new detections
        tracked_detections, alert_active = self.tracker.update(
            raw_detections, timestamp
        )

        # Step 3: Update FPS counter
        fps = self.fps_counter.tick()

        # Step 4: Build annotated frame
        annotated = self._annotate_frame(
            frame, tracked_detections, alert_active, fps, frame_id
        )

        # Step 5: Get tracker status
        status = self.tracker.get_status()
        best = self.tracker.get_best_detection()

        # Step 6: Build result
        result = {
            "phone_detected": status["phone_detected"],
            "confidence": round(status["confidence"], 4),
            "timestamp": format_timestamp(timestamp),
            "frame_id": frame_id,
            "alert_active": alert_active,
            "detections": [
                {
                    "bbox": d["bbox"],
                    "confidence": round(d["confidence"], 4),
                    "track_id": d.get("track_id"),
                    "consecutive_frames": d.get("consecutive_frames", 0),
                    "avg_confidence": round(d.get("avg_confidence", d["confidence"]), 4),
                }
                for d in tracked_detections
            ],
            "fps": round(fps, 1),
            "annotated_frame": annotated,
        }

        return result

    def _annotate_frame(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        alert_active: bool,
        fps: float,
        frame_id: int,
    ) -> np.ndarray:
        """
        Build a fully annotated frame for display/saving.

        Args:
            frame: Raw input frame
            detections: Tracked detection list
            alert_active: Whether alert is active
            fps: Current FPS
            frame_id: Current frame index

        Returns:
            Annotated frame with all overlays
        """
        # Draw detection bounding boxes
        annotated = draw_detections(frame, detections, is_alert=alert_active)

        # Draw alert banner if phone is persistently detected
        if alert_active:
            best_conf = max(
                (d.get("avg_confidence", d["confidence"]) for d in detections),
                default=0.0
            )
            annotated = draw_alert_banner(annotated, confidence=best_conf)

        # Draw FPS counter
        annotated = draw_fps_counter(annotated, fps, position="bottom-left")

        # Draw status panel
        status = self.tracker.get_status()
        status["frame_id"] = frame_id
        annotated = draw_status_panel(annotated, status)

        return annotated

    def process_video(
        self,
        video_path: str,
        output_path: str = None,
        skip_frames: int = 0,
    ) -> Dict:
        """
        Process a complete video file.

        Args:
            video_path: Path to input video file
            output_path: Path to save annotated output video (auto-generated if None)
            skip_frames: Number of frames to skip between processed frames
                         (0 = process all frames, 1 = skip every other, etc.)

        Returns:
            Summary dictionary with detection statistics
        """
        if not validate_video_file(video_path):
            raise FileNotFoundError(f"Invalid or unsupported video file: {video_path}")

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(f"Video: {video_path}")
        logger.info(f"  Resolution: {frame_w}x{frame_h}")
        logger.info(f"  FPS: {fps}")
        logger.info(f"  Total frames: {total_frames}")

        # Set up output video writer
        writer = None
        if self.save_output:
            if output_path is None:
                stem = Path(video_path).stem
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = str(OUTPUTS_DIR / f"{stem}_detected_{ts}.mp4")

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))
            logger.info(f"Output video: {output_path}")

        # Reset tracker for new video
        self.tracker.reset()

        # Processing loop
        all_results = []
        frame_id = 0
        processed_count = 0
        start_time = time.time()
        skip_counter = 0

        logger.info("Processing video...")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_id += 1

                # Handle frame skipping
                if skip_counter > 0:
                    skip_counter -= 1
                    if writer and frame_id > 1:
                        writer.write(frame)  # Write unprocessed frame as-is
                    continue
                skip_counter = skip_frames

                timestamp = frame_id / fps
                result = self.process_frame(frame, frame_id, timestamp)
                annotated = result.pop("annotated_frame")
                all_results.append(result)
                processed_count += 1

                # Save annotated frame
                if writer:
                    writer.write(annotated)

                # Show display
                if self.show_display:
                    cv2.imshow("Phone Detection - Video", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if cv2.getWindowProperty("Phone Detection - Video", cv2.WND_PROP_VISIBLE) < 1:
                        logger.info("Window closed by user")
                        break
                    if key == ord("q"):
                        logger.info("Stopped by user (q key)")
                        break

                # Progress log every 100 frames
                if frame_id % 100 == 0:
                    elapsed = time.time() - start_time
                    pct = (frame_id / total_frames * 100) if total_frames > 0 else 0
                    logger.info(
                        f"  Progress: {pct:.1f}% | "
                        f"Frame {frame_id}/{total_frames} | "
                        f"FPS: {processed_count / elapsed:.1f}"
                    )

        finally:
            cap.release()
            if writer:
                writer.release()
            if self.show_display:
                cv2.destroyAllWindows()

        # Build summary
        total_time = time.time() - start_time
        avg_fps = processed_count / total_time if total_time > 0 else 0
        alert_frames = sum(1 for r in all_results if r["alert_active"])
        max_conf = max((r["confidence"] for r in all_results), default=0.0)
        phone_detected_frames = sum(1 for r in all_results if r["phone_detected"])

        summary = {
            "video_path": video_path,
            "output_path": output_path,
            "total_frames": frame_id,
            "processed_frames": processed_count,
            "processing_time_sec": round(total_time, 2),
            "avg_fps": round(avg_fps, 1),
            "phone_detected_frames": phone_detected_frames,
            "alert_frames": alert_frames,
            "max_confidence": round(max_conf, 4),
            "total_alerts": self.tracker.alert_count,
            "suspicious_timestamps": self.tracker.suspicious_timestamps,
        }

        logger.info("=" * 50)
        logger.info("VIDEO PROCESSING COMPLETE")
        logger.info(f"  Processed frames:   {processed_count}")
        logger.info(f"  Avg FPS:            {avg_fps:.1f}")
        logger.info(f"  Phone detected in:  {phone_detected_frames} frames")
        logger.info(f"  Alert frames:       {alert_frames}")
        logger.info(f"  Total alerts:       {self.tracker.alert_count}")
        logger.info(f"  Max confidence:     {max_conf:.2%}")
        logger.info("=" * 50)

        # Save detection log CSV
        if self.tracker.suspicious_timestamps:
            self._save_suspicious_log(video_path)

        return summary

    def process_webcam(
        self,
        camera_id: int = 0,
        output_path: str = None,
        duration_sec: float = None,
    ) -> Dict:
        """
        Process live webcam stream.

        Args:
            camera_id: Camera device ID (0 = default webcam)
            output_path: Optional path to save output video
            duration_sec: Optional max duration in seconds (None = run until 'q')

        Returns:
            Summary dictionary
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open webcam (camera_id={camera_id}). "
                "Ensure webcam is connected and not in use."
            )

        # Get webcam properties
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(f"Webcam opened: {frame_w}x{frame_h}")
        logger.info("Press 'q' to quit | 's' to save snapshot | 'r' to reset tracker")

        # Set up output writer
        writer = None
        if self.save_output and output_path:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_file = output_path or str(OUTPUTS_DIR / f"webcam_{ts}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_file, fourcc, 20, (frame_w, frame_h))

        # Reset tracker
        self.tracker.reset()

        all_results = []
        frame_id = 0
        start_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame from webcam")
                    break

                frame_id += 1
                elapsed = time.time() - start_time
                timestamp = elapsed

                # Check duration limit
                if duration_sec and elapsed >= duration_sec:
                    logger.info(f"Duration limit reached: {duration_sec}s")
                    break

                result = self.process_frame(frame, frame_id, timestamp)
                annotated = result.pop("annotated_frame")
                all_results.append(result)

                if writer:
                    writer.write(annotated)

                # Display
                cv2.imshow("Phone Detection - Live Webcam", annotated)
                key = cv2.waitKey(1) & 0xFF

                # Check if window X button was clicked
                if cv2.getWindowProperty("Phone Detection - Live Webcam", cv2.WND_PROP_VISIBLE) < 1:
                    logger.info("Window closed by user")
                    break

                if key == ord("q"):
                    logger.info("Stopped by user (q key)")
                    break
                elif key == ord("s"):
                    snap_path = str(OUTPUTS_DIR / f"snapshot_{frame_id}.jpg")
                    cv2.imwrite(snap_path, annotated)
                    logger.info(f"Snapshot saved: {snap_path}")
                elif key == ord("r"):
                    self.tracker.reset()
                    logger.info("Tracker reset")

        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()

        # Build summary
        total_time = time.time() - start_time
        phone_frames = sum(1 for r in all_results if r["phone_detected"])
        summary = {
            "source": f"webcam:{camera_id}",
            "total_frames": frame_id,
            "total_time_sec": round(total_time, 2),
            "phone_detected_frames": phone_frames,
            "total_alerts": self.tracker.alert_count,
            "suspicious_timestamps": self.tracker.suspicious_timestamps,
        }

        return summary

    def _save_suspicious_log(self, source_path: str) -> str:
        """
        Save suspicious detection timestamps to a CSV log file.

        Args:
            source_path: Original video path (used for naming)

        Returns:
            Path to saved CSV file
        """
        stem = Path(source_path).stem
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = OUTPUTS_DIR / f"{stem}_alerts_{ts}.csv"

        fieldnames = [
            "alert_number", "timestamp", "frame_id",
            "track_id", "consecutive_frames", "avg_confidence"
        ]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.tracker.suspicious_timestamps)

        logger.info(f"Alert log saved: {csv_path}")
        return str(csv_path)
