"""
==============================================================
Helper Utilities - Mobile Phone Detection System
==============================================================
General-purpose helper functions for device detection,
timestamp formatting, FPS calculation, and file validation.
==============================================================
"""

import time
from pathlib import Path
from collections import deque

import torch
import numpy as np

from config import ALLOWED_EXTENSIONS


def get_device(preferred: str = "auto") -> str:
    """
    Determine the best available compute device.

    Args:
        preferred: Preferred device ('auto', 'cpu', 'cuda', 'mps')

    Returns:
        Device string compatible with Ultralytics YOLO
    """
    if preferred == "auto":
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"[✓] GPU Detected: {device_name}")
            print(f"    CUDA Version: {torch.version.cuda}")
            print(f"    GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("[✓] Apple MPS (Metal) detected")
            return "mps"
        else:
            print("[!] No GPU found, using CPU")
            return "cpu"
    return preferred


def format_timestamp(seconds: float) -> str:
    """
    Convert seconds to HH:MM:SS.mmm format.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted timestamp string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


class FPSCounter:
    """
    Smooth FPS counter using a rolling window.
    Provides accurate FPS measurement with temporal smoothing.
    """

    def __init__(self, window_size: int = 30):
        """
        Initialize FPS counter.

        Args:
            window_size: Number of frames to average over
        """
        self.window_size = window_size
        self.timestamps = deque(maxlen=window_size)
        self.fps = 0.0

    def tick(self) -> float:
        """
        Record a frame timestamp and calculate FPS.

        Returns:
            Current smoothed FPS value
        """
        now = time.time()
        self.timestamps.append(now)

        if len(self.timestamps) >= 2:
            elapsed = self.timestamps[-1] - self.timestamps[0]
            if elapsed > 0:
                self.fps = (len(self.timestamps) - 1) / elapsed

        return self.fps

    def get_fps(self) -> float:
        """Get current FPS value."""
        return self.fps


def calculate_fps(start_time: float, frame_count: int) -> float:
    """
    Calculate average FPS from start time and frame count.

    Args:
        start_time: Processing start time
        frame_count: Total frames processed

    Returns:
        Average FPS
    """
    elapsed = time.time() - start_time
    if elapsed > 0:
        return frame_count / elapsed
    return 0.0


def validate_video_file(filepath: str) -> bool:
    """
    Validate that a file is a supported video format.

    Args:
        filepath: Path to the video file

    Returns:
        True if valid video file
    """
    path = Path(filepath)
    return path.exists() and path.suffix.lower() in ALLOWED_EXTENSIONS


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: First bounding box [x1, y1, x2, y2]
        box2: Second bounding box [x1, y1, x2, y2]

    Returns:
        IoU score between 0 and 1
    """
    # Intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


def get_bbox_area(bbox: np.ndarray) -> float:
    """
    Calculate bounding box area.

    Args:
        bbox: Bounding box [x1, y1, x2, y2]

    Returns:
        Area in pixels²
    """
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def get_aspect_ratio(bbox: np.ndarray) -> float:
    """
    Calculate bounding box aspect ratio (width / height).

    Args:
        bbox: Bounding box [x1, y1, x2, y2]

    Returns:
        Aspect ratio (width / height)
    """
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    if height == 0:
        return 0.0
    return width / height
