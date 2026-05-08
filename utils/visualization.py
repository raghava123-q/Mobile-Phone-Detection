"""
==============================================================
Visualization Module - Mobile Phone Detection System
==============================================================
OpenCV-based visualization utilities for drawing bounding boxes,
alert banners, FPS counters, and detection overlays on frames.
==============================================================
"""

import cv2
import numpy as np

from config import (
    BBOX_COLOR_ALERT,
    BBOX_COLOR_TRACKING,
    BBOX_COLOR_NORMAL,
    ALERT_BANNER_HEIGHT,
    ALERT_BANNER_COLOR,
    ALERT_TEXT_COLOR,
    FONT_SCALE,
    FONT_THICKNESS,
)


def draw_detections(
    frame: np.ndarray,
    detections: list,
    is_alert: bool = False,
) -> np.ndarray:
    """
    Draw bounding boxes and labels for all detections on the frame.

    Each detection dict should contain:
        - bbox: [x1, y1, x2, y2]
        - confidence: float
        - class_name: str
        - persistent_frames: int (optional)
        - track_id: int (optional)

    Args:
        frame: Input frame (BGR)
        detections: List of detection dictionaries
        is_alert: Whether an alert is currently active

    Returns:
        Annotated frame
    """
    annotated = frame.copy()

    for det in detections:
        bbox = det["bbox"]
        conf = det["confidence"]
        class_name = det.get("class_name", "cell phone")
        persistent = det.get("persistent_frames", 0)
        track_id = det.get("track_id", None)

        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        # Select color based on state
        if is_alert:
            color = BBOX_COLOR_ALERT
        elif persistent > 0:
            color = BBOX_COLOR_TRACKING
        else:
            color = BBOX_COLOR_NORMAL

        # Draw bounding box with rounded corners effect
        thickness = 3 if is_alert else 2
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # Draw corner accents for premium look
        corner_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
        cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), color, thickness + 1)
        cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), color, thickness + 1)
        cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), color, thickness + 1)
        cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), color, thickness + 1)
        cv2.line(annotated, (x1, y2), (x1 + corner_len, y2), color, thickness + 1)
        cv2.line(annotated, (x1, y2), (x1, y2 - corner_len), color, thickness + 1)
        cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), color, thickness + 1)
        cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), color, thickness + 1)

        # Build label text
        label_parts = [f"{class_name}: {conf:.0%}"]
        if track_id is not None:
            label_parts.insert(0, f"ID:{track_id}")
        if persistent > 0:
            label_parts.append(f"[{persistent}F]")
        label = " | ".join(label_parts)

        # Draw label background
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS
        )
        label_y = max(y1 - 10, label_h + 10)

        cv2.rectangle(
            annotated,
            (x1, label_y - label_h - 8),
            (x1 + label_w + 8, label_y + 4),
            color,
            -1,
        )

        # Draw label text
        cv2.putText(
            annotated,
            label,
            (x1 + 4, label_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            (255, 255, 255),
            FONT_THICKNESS,
            cv2.LINE_AA,
        )

    return annotated


def draw_alert_banner(
    frame: np.ndarray,
    message: str = "⚠ PHONE DETECTED - CHEATING ALERT ⚠",
    confidence: float = 0.0,
) -> np.ndarray:
    """
    Draw a red alert banner at the top of the frame.

    Args:
        frame: Input frame (BGR)
        message: Alert message text
        confidence: Detection confidence for display

    Returns:
        Frame with alert banner overlay
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Create semi-transparent red banner
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, ALERT_BANNER_HEIGHT), ALERT_BANNER_COLOR, -1)
    cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)

    # Pulsating border effect (simulated with thick border)
    cv2.rectangle(annotated, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)

    # Draw alert text
    text = f"{message}  |  Confidence: {confidence:.0%}"
    (text_w, text_h), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
    )
    text_x = (w - text_w) // 2
    text_y = (ALERT_BANNER_HEIGHT + text_h) // 2

    cv2.putText(
        annotated,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        ALERT_TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )

    return annotated


def draw_fps_counter(
    frame: np.ndarray,
    fps: float,
    position: str = "bottom-left",
) -> np.ndarray:
    """
    Draw FPS counter on the frame.

    Args:
        frame: Input frame (BGR)
        fps: Current FPS value
        position: Counter position ('top-left', 'top-right', 'bottom-left', 'bottom-right')

    Returns:
        Frame with FPS counter
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    fps_text = f"FPS: {fps:.1f}"
    (text_w, text_h), _ = cv2.getTextSize(
        fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    )

    padding = 8
    positions = {
        "top-left": (padding, text_h + padding + 5),
        "top-right": (w - text_w - padding - 5, text_h + padding + 5),
        "bottom-left": (padding, h - padding - 5),
        "bottom-right": (w - text_w - padding - 5, h - padding - 5),
    }

    x, y = positions.get(position, positions["bottom-left"])

    # Draw background rectangle
    cv2.rectangle(
        annotated,
        (x - padding, y - text_h - padding),
        (x + text_w + padding, y + padding),
        (0, 0, 0),
        -1,
    )

    # Choose color based on FPS
    if fps >= 25:
        color = (0, 255, 0)    # Green - good
    elif fps >= 15:
        color = (0, 200, 255)  # Yellow - okay
    else:
        color = (0, 0, 255)    # Red - slow

    cv2.putText(
        annotated,
        fps_text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )

    return annotated


def draw_status_panel(
    frame: np.ndarray,
    status: dict,
) -> np.ndarray:
    """
    Draw a status information panel on the frame.

    Args:
        frame: Input frame (BGR)
        status: Dictionary with status information:
            - phone_detected: bool
            - confidence: float
            - tracked_objects: int
            - frame_id: int
            - alert_active: bool

    Returns:
        Frame with status panel
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Panel dimensions
    panel_w = 280
    panel_h = 160
    panel_x = w - panel_w - 10
    panel_y = h - panel_h - 10

    # Draw semi-transparent panel background
    overlay = annotated.copy()
    cv2.rectangle(
        overlay,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h),
        (30, 30, 30),
        -1,
    )
    cv2.addWeighted(overlay, 0.8, annotated, 0.2, 0, annotated)

    # Draw panel border
    border_color = (0, 0, 255) if status.get("alert_active", False) else (100, 100, 100)
    cv2.rectangle(
        annotated,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h),
        border_color,
        2,
    )

    # Draw title
    cv2.putText(
        annotated,
        "DETECTION STATUS",
        (panel_x + 10, panel_y + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    # Draw separator line
    cv2.line(
        annotated,
        (panel_x + 10, panel_y + 35),
        (panel_x + panel_w - 10, panel_y + 35),
        (80, 80, 80),
        1,
    )

    # Status lines
    lines = [
        (
            f"Phone: {'DETECTED' if status.get('phone_detected') else 'None'}",
            (0, 0, 255) if status.get("phone_detected") else (0, 255, 0),
        ),
        (
            f"Confidence: {status.get('confidence', 0):.0%}",
            (255, 255, 255),
        ),
        (
            f"Tracked: {status.get('tracked_objects', 0)} objects",
            (255, 255, 255),
        ),
        (
            f"Frame: #{status.get('frame_id', 0)}",
            (180, 180, 180),
        ),
    ]

    for i, (text, color) in enumerate(lines):
        cv2.putText(
            annotated,
            text,
            (panel_x + 15, panel_y + 60 + i * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    return annotated
