"""
==============================================================
Export Utilities - Mobile Phone Detection System
==============================================================
Export detection logs and alerts to CSV for analysis.
==============================================================
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict

from config import OUTPUTS_DIR
from utils.logger import setup_logger

logger = setup_logger("Export")


def export_alerts_csv(alerts: List[Dict], filename: str = None) -> str:
    """
    Export suspicious alert timestamps to a CSV file.

    Args:
        alerts: List of alert dictionaries from the tracker
        filename: Optional output filename

    Returns:
        Path to the saved CSV file
    """
    if filename is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detection_alerts_{ts}.csv"

    filepath = OUTPUTS_DIR / filename
    fieldnames = [
        "alert_number", "timestamp", "frame_id",
        "track_id", "consecutive_frames", "avg_confidence"
    ]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for alert in alerts:
            row = {k: alert.get(k, "") for k in fieldnames}
            writer.writerow(row)

    logger.info(f"Alerts exported to CSV: {filepath}")
    return str(filepath)


def export_detection_history_csv(
    history: List[Dict],
    filename: str = None
) -> str:
    """
    Export frame-by-frame detection history to CSV.

    Args:
        history: Detection history from the tracker
        filename: Optional output filename

    Returns:
        Path to the saved CSV file
    """
    if filename is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detection_history_{ts}.csv"

    filepath = OUTPUTS_DIR / filename
    fieldnames = [
        "frame_id", "timestamp", "alert_active",
        "detection_count", "phone_detected"
    ]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in history:
            writer.writerow({
                "frame_id": record.get("frame_id", 0),
                "timestamp": record.get("timestamp", 0),
                "alert_active": record.get("alert_active", False),
                "detection_count": len(record.get("detections", [])),
                "phone_detected": len(record.get("detections", [])) > 0,
            })

    logger.info(f"History exported to CSV: {filepath}")
    return str(filepath)


def export_results_json(results: Dict, filename: str = None) -> str:
    """
    Export detection results to JSON file.

    Args:
        results: Detection results dictionary
        filename: Optional output filename

    Returns:
        Path to the saved JSON file
    """
    if filename is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detection_results_{ts}.json"

    filepath = OUTPUTS_DIR / filename

    # Clean non-serializable values
    clean = {}
    for key, value in results.items():
        if key == "annotated_frame":
            continue
        clean[key] = value

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, default=str)

    logger.info(f"Results exported to JSON: {filepath}")
    return str(filepath)
