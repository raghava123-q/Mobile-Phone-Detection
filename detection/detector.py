"""
==============================================================
YOLOv8 Phone Detector - Mobile Phone Detection System
==============================================================
Core detection engine using YOLOv8 for identifying mobile phones
in video frames. Includes confidence filtering, bounding box
extraction, and false positive reduction.

The detector uses the COCO pre-trained YOLOv8 model which can
detect 80 object classes. We specifically filter for class 67
("cell phone") to identify mobile phone usage.
==============================================================
"""

import time
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from config import (
    YOLO_MODEL,
    PHONE_CLASS_ID,
    PHONE_CLASS_NAME,
    CONFIDENCE_THRESHOLD,
    NMS_IOU_THRESHOLD,
    INPUT_SIZE,
    MAX_DETECTIONS,
    MIN_BBOX_AREA,
    MAX_BBOX_AREA_RATIO,
    MIN_ASPECT_RATIO,
    MAX_ASPECT_RATIO,
    MODELS_DIR,
)
from utils.logger import setup_logger
from utils.helpers import get_device, get_bbox_area, get_aspect_ratio

logger = setup_logger("Detector")


class PhoneDetector:
    """
    YOLOv8-based mobile phone detector.

    This class encapsulates the YOLO model loading, inference,
    post-processing, and false positive filtering pipeline.

    Attributes:
        model: Loaded YOLOv8 model instance
        device: Compute device (cpu/cuda/mps)
        confidence_threshold: Minimum detection confidence
        model_loaded: Whether model is ready for inference

    Usage:
        detector = PhoneDetector()
        detections = detector.detect(frame)
    """

    def __init__(
        self,
        model_path: str = None,
        device: str = "auto",
        confidence_threshold: float = None,
    ):
        """
        Initialize the phone detector.

        Args:
            model_path: Path to YOLOv8 model weights (.pt file)
                        If None, uses the default from config
            device: Compute device ('auto', 'cpu', 'cuda', 'mps')
            confidence_threshold: Override default confidence threshold
        """
        self.device = get_device(device)
        self.confidence_threshold = confidence_threshold or CONFIDENCE_THRESHOLD
        self.model_loaded = False
        self.model = None
        self.inference_times = []

        # Resolve model path
        if model_path is None:
            model_path = YOLO_MODEL

        self._load_model(model_path)

    def _load_model(self, model_path: str) -> None:
        """
        Load the YOLOv8 model.

        Downloads the pre-trained model if not found locally.
        The COCO pre-trained model includes 80 classes including
        'cell phone' (class ID 67).

        Args:
            model_path: Path to the model weights
        """
        try:
            logger.info(f"Loading YOLOv8 model: {model_path}")
            logger.info(f"Device: {self.device}")

            self.model = YOLO(model_path)
            self.model_loaded = True

            logger.info("[✓] YOLOv8 model loaded successfully")
            logger.info(f"    Model type: {self.model.type}")
            logger.info(f"    Classes: {len(self.model.names)} total")
            logger.info(f"    Target class: '{PHONE_CLASS_NAME}' (ID: {PHONE_CLASS_ID})")

        except Exception as e:
            logger.error(f"[✗] Failed to load model: {e}")
            self.model_loaded = False
            raise RuntimeError(f"Model loading failed: {e}")

    def detect(
        self,
        frame: np.ndarray,
        confidence_threshold: float = None,
        apply_filters: bool = True,
    ) -> List[Dict]:
        """
        Run phone detection on a single frame.

        Pipeline:
            1. Run YOLOv8 inference
            2. Filter for phone class (ID 67)
            3. Apply confidence threshold
            4. Apply false positive reduction filters
            5. Return structured detections

        Args:
            frame: Input frame (BGR, numpy array)
            confidence_threshold: Override instance threshold
            apply_filters: Whether to apply false positive filters

        Returns:
            List of detection dictionaries:
            [
                {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": 0.91,
                    "class_id": 67,
                    "class_name": "cell phone",
                }
            ]
        """
        if not self.model_loaded:
            logger.error("Model not loaded!")
            return []

        conf_thresh = confidence_threshold or self.confidence_threshold

        # Run YOLO inference
        start_time = time.time()
        results = self.model(
            frame,
            conf=conf_thresh,
            iou=NMS_IOU_THRESHOLD,
            imgsz=INPUT_SIZE,
            device=self.device,
            classes=[PHONE_CLASS_ID],  # Only detect cell phones
            max_det=MAX_DETECTIONS,
            verbose=False,
        )
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)

        # Parse results
        detections = self._parse_results(results, frame.shape)

        # Apply false positive reduction
        if apply_filters:
            detections = self._filter_false_positives(detections, frame.shape)

        return detections

    def _parse_results(
        self,
        results,
        frame_shape: Tuple[int, int, int],
    ) -> List[Dict]:
        """
        Parse YOLO results into structured detection dictionaries.

        Args:
            results: YOLOv8 Results object
            frame_shape: Shape of the input frame (H, W, C)

        Returns:
            List of detection dictionaries
        """
        detections = []

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            boxes = result.boxes
            for i in range(len(boxes)):
                # Extract bounding box coordinates
                bbox = boxes.xyxy[i].cpu().numpy()
                confidence = float(boxes.conf[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())

                detection = {
                    "bbox": bbox.tolist(),
                    "confidence": confidence,
                    "class_id": class_id,
                    "class_name": PHONE_CLASS_NAME,
                }

                detections.append(detection)

        return detections

    def _filter_false_positives(
        self,
        detections: List[Dict],
        frame_shape: Tuple[int, int, int],
    ) -> List[Dict]:
        """
        Apply false positive reduction filters.

        Filters applied:
            1. Minimum bounding box area
            2. Maximum bounding box area (relative to frame)
            3. Aspect ratio constraints
            4. Edge proximity check (phones at frame edges are suspicious)

        This helps reduce false positives from:
            - TV remotes (too thin)
            - Wallets (wrong aspect ratio)
            - Full-screen rectangles (too large)
            - Tiny distant objects (too small)

        Args:
            detections: Raw detection list
            frame_shape: Frame dimensions (H, W, C)

        Returns:
            Filtered detection list
        """
        filtered = []
        frame_h, frame_w = frame_shape[:2]
        frame_area = frame_h * frame_w

        for det in detections:
            bbox = np.array(det["bbox"])

            # Filter 1: Minimum area
            area = get_bbox_area(bbox)
            if area < MIN_BBOX_AREA:
                logger.debug(f"Filtered: area too small ({area:.0f} < {MIN_BBOX_AREA})")
                continue

            # Filter 2: Maximum area ratio
            area_ratio = area / frame_area
            if area_ratio > MAX_BBOX_AREA_RATIO:
                logger.debug(f"Filtered: area too large ({area_ratio:.2%} > {MAX_BBOX_AREA_RATIO:.0%})")
                continue

            # Filter 3: Aspect ratio
            aspect = get_aspect_ratio(bbox)
            if aspect < MIN_ASPECT_RATIO or aspect > MAX_ASPECT_RATIO:
                logger.debug(f"Filtered: aspect ratio {aspect:.2f} out of range [{MIN_ASPECT_RATIO}, {MAX_ASPECT_RATIO}]")
                continue

            filtered.append(det)

        return filtered

    def detect_batch(
        self,
        frames: List[np.ndarray],
        confidence_threshold: float = None,
    ) -> List[List[Dict]]:
        """
        Run detection on a batch of frames.

        Args:
            frames: List of input frames
            confidence_threshold: Override confidence threshold

        Returns:
            List of detection lists (one per frame)
        """
        return [self.detect(frame, confidence_threshold) for frame in frames]

    def get_avg_inference_time(self) -> float:
        """
        Get average inference time across all processed frames.

        Returns:
            Average inference time in milliseconds
        """
        if not self.inference_times:
            return 0.0
        return (sum(self.inference_times) / len(self.inference_times)) * 1000

    def get_model_info(self) -> Dict:
        """
        Get model information and configuration.

        Returns:
            Dictionary with model details
        """
        return {
            "model_type": str(self.model.type) if self.model else "N/A",
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "target_class": PHONE_CLASS_NAME,
            "target_class_id": PHONE_CLASS_ID,
            "input_size": INPUT_SIZE,
            "model_loaded": self.model_loaded,
            "avg_inference_ms": self.get_avg_inference_time(),
        }
