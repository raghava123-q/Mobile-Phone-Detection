"""
==============================================================
Persistence Tracker - Mobile Phone Detection System
==============================================================
Frame-level tracking logic that:
  - Assigns track IDs to detections across frames
  - Counts consecutive frames a phone has been visible
  - Applies temporal smoothing via confidence averaging
  - Triggers alerts after N consecutive frames
  - Implements alert cooldown mechanism
  - Maintains a detection history buffer

How Persistence Tracking Works:
  1. Each new detection is matched to existing tracks using IoU
  2. Matched tracks increment their frame counter
  3. Unmatched tracks increment missing frame counter
  4. Tracks with enough consecutive frames → cheating alert
  5. Alert cooldown prevents repeated notifications
==============================================================
"""

import time
from collections import deque
from typing import List, Dict, Optional, Tuple

import numpy as np

from config import (
    PERSISTENCE_THRESHOLD,
    MAX_MISSING_FRAMES,
    TRACKING_IOU_THRESHOLD,
    CONFIDENCE_WINDOW_SIZE,
    ALERT_COOLDOWN_SECONDS,
    ALERT_MIN_AVG_CONFIDENCE,
)
from utils.logger import setup_logger
from utils.helpers import calculate_iou

logger = setup_logger("Tracker")


class TrackedObject:
    """
    Represents a single tracked phone detection across frames.

    Maintains bounding box history, confidence scores,
    and persistence counters for each tracked object.
    """

    _id_counter = 0

    def __init__(self, bbox: List[float], confidence: float):
        """
        Initialize a new tracked object.

        Args:
            bbox: Initial bounding box [x1, y1, x2, y2]
            confidence: Initial detection confidence
        """
        TrackedObject._id_counter += 1
        self.track_id = TrackedObject._id_counter

        self.bbox = bbox
        self.confidence = confidence

        # Persistence counters
        self.consecutive_frames = 1      # Consecutive frames detected
        self.missing_frames = 0          # Consecutive frames NOT detected
        self.total_frames = 1            # Total frames this track has been active

        # Confidence history for temporal smoothing
        self.confidence_history = deque(maxlen=CONFIDENCE_WINDOW_SIZE)
        self.confidence_history.append(confidence)

        # Track state
        self.is_active = True
        self.alert_triggered = False
        self.first_seen = time.time()
        self.last_seen = time.time()

    def update(self, bbox: List[float], confidence: float) -> None:
        """
        Update track with a new matching detection.

        Applies exponential smoothing to bbox coordinates for
        stable visualization, and tracks confidence history.

        Args:
            bbox: New bounding box [x1, y1, x2, y2]
            confidence: New detection confidence
        """
        # Smooth bounding box with exponential moving average (alpha=0.6)
        alpha = 0.6
        self.bbox = [
            alpha * bbox[i] + (1 - alpha) * self.bbox[i]
            for i in range(4)
        ]

        self.confidence = confidence
        self.confidence_history.append(confidence)
        self.consecutive_frames += 1
        self.missing_frames = 0
        self.total_frames += 1
        self.last_seen = time.time()

    def mark_missing(self) -> None:
        """Mark this frame as missing (no matching detection found)."""
        self.consecutive_frames = 0
        self.missing_frames += 1
        self.total_frames += 1

    def get_avg_confidence(self) -> float:
        """
        Get temporally smoothed confidence from recent history.

        Returns:
            Average confidence over the sliding window
        """
        if not self.confidence_history:
            return 0.0
        return sum(self.confidence_history) / len(self.confidence_history)

    def should_alert(self) -> bool:
        """
        Determine if this track should trigger a cheating alert.

        Alert conditions:
            - Phone visible for >= PERSISTENCE_THRESHOLD consecutive frames
            - Average confidence >= ALERT_MIN_AVG_CONFIDENCE

        Returns:
            True if alert should be triggered
        """
        return (
            self.consecutive_frames >= PERSISTENCE_THRESHOLD
            and self.get_avg_confidence() >= ALERT_MIN_AVG_CONFIDENCE
        )

    def to_dict(self) -> Dict:
        """
        Serialize the tracked object to a dictionary.

        Returns:
            Dictionary representation of the track
        """
        return {
            "track_id": self.track_id,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "avg_confidence": self.get_avg_confidence(),
            "consecutive_frames": self.consecutive_frames,
            "missing_frames": self.missing_frames,
            "total_frames": self.total_frames,
            "is_active": self.is_active,
            "alert_triggered": self.alert_triggered,
            "class_name": "cell phone",
        }


class PersistenceTracker:
    """
    Multi-object tracker with persistence-based cheating detection.

    This tracker maintains a list of active phone tracks, matches
    new detections to existing tracks using IoU, and fires alerts
    when a phone is persistently detected across frames.

    Features:
        - IoU-based detection-to-track matching
        - Temporal confidence smoothing
        - Consecutive frame persistence counting
        - Alert cooldown mechanism
        - Detection history buffer
        - Track lifecycle management
    """

    def __init__(self):
        """Initialize the persistence tracker."""
        self.tracks: List[TrackedObject] = []
        self.frame_count = 0

        # Alert management
        self.alert_active = False
        self.last_alert_time = 0.0
        self.alert_count = 0

        # Detection history buffer (frame_id → detections)
        self.detection_history: deque = deque(maxlen=200)

        # Suspicious timestamps log
        self.suspicious_timestamps: List[Dict] = []

        logger.info("[✓] PersistenceTracker initialized")
        logger.info(f"    Persistence threshold: {PERSISTENCE_THRESHOLD} frames")
        logger.info(f"    Alert cooldown: {ALERT_COOLDOWN_SECONDS}s")
        logger.info(f"    IoU threshold: {TRACKING_IOU_THRESHOLD}")

    def update(
        self,
        detections: List[Dict],
        timestamp: float = None,
    ) -> Tuple[List[Dict], bool]:
        """
        Update tracker state with new frame detections.

        Algorithm:
            1. Match current detections to existing tracks via IoU
            2. Update matched tracks with new positions/confidence
            3. Create new tracks for unmatched detections
            4. Mark unmatched tracks as missing
            5. Remove stale tracks (too many missing frames)
            6. Evaluate alert conditions

        Args:
            detections: List of detection dicts from the detector
            timestamp: Current video timestamp (seconds)

        Returns:
            Tuple of:
                - Enhanced detection list with tracking info
                - Boolean: whether alert is currently active
        """
        self.frame_count += 1
        if timestamp is None:
            timestamp = time.time()

        # Step 1: Match detections to existing active tracks
        active_tracks = [t for t in self.tracks if t.is_active]
        matched_track_ids = set()
        matched_det_indices = set()

        if active_tracks and detections:
            iou_matrix = self._compute_iou_matrix(active_tracks, detections)

            # Greedy matching (highest IoU first)
            while True:
                if iou_matrix.size == 0:
                    break
                max_iou = iou_matrix.max()
                if max_iou < TRACKING_IOU_THRESHOLD:
                    break

                track_idx, det_idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
                track = active_tracks[track_idx]
                det = detections[det_idx]

                track.update(det["bbox"], det["confidence"])
                matched_track_ids.add(track.track_id)
                matched_det_indices.add(det_idx)

                # Zero out matched row and column to prevent re-matching
                iou_matrix[track_idx, :] = 0
                iou_matrix[:, det_idx] = 0

        # Step 2: Mark unmatched tracks as missing
        for track in active_tracks:
            if track.track_id not in matched_track_ids:
                track.mark_missing()
                if track.missing_frames > MAX_MISSING_FRAMES:
                    track.is_active = False
                    logger.debug(f"Track {track.track_id} removed (too many missing frames)")

        # Step 3: Create new tracks for unmatched detections
        for idx, det in enumerate(detections):
            if idx not in matched_det_indices:
                new_track = TrackedObject(det["bbox"], det["confidence"])
                self.tracks.append(new_track)
                logger.debug(f"New track created: ID={new_track.track_id}, conf={det['confidence']:.2f}")

        # Step 4: Evaluate alerts
        self.alert_active = self._evaluate_alert(timestamp)

        # Step 5: Build enriched detection list
        enriched_detections = self._build_enriched_detections()

        # Step 6: Record in history
        self.detection_history.append({
            "frame_id": self.frame_count,
            "timestamp": timestamp,
            "detections": enriched_detections,
            "alert_active": self.alert_active,
        })

        return enriched_detections, self.alert_active

    def _compute_iou_matrix(
        self,
        tracks: List[TrackedObject],
        detections: List[Dict],
    ) -> np.ndarray:
        """
        Compute IoU matrix between all tracks and detections.

        Args:
            tracks: List of active tracked objects
            detections: List of current detections

        Returns:
            IoU matrix of shape (n_tracks, n_detections)
        """
        matrix = np.zeros((len(tracks), len(detections)))
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                iou = calculate_iou(
                    np.array(track.bbox),
                    np.array(det["bbox"])
                )
                matrix[i, j] = iou
        return matrix

    def _evaluate_alert(self, timestamp: float) -> bool:
        """
        Evaluate whether a cheating alert should be active.

        Checks all active tracks for persistence threshold breach
        while respecting the cooldown period.

        Args:
            timestamp: Current timestamp

        Returns:
            True if alert should be active
        """
        now = time.time()
        cooldown_ok = (now - self.last_alert_time) >= ALERT_COOLDOWN_SECONDS
        any_alert = False

        for track in self.tracks:
            if not track.is_active:
                continue

            # Reset alert_triggered after cooldown so it can re-trigger
            if track.alert_triggered and cooldown_ok:
                track.alert_triggered = False

            if track.should_alert():
                any_alert = True  # Keep alert banner visible

                if cooldown_ok and not track.alert_triggered:
                    # Log a new alert event
                    self.alert_count += 1
                    self.last_alert_time = now
                    track.alert_triggered = True
                    logger.warning(
                        f"[ALERT] Phone detected! Track ID={track.track_id}, "
                        f"Frames={track.consecutive_frames}, "
                        f"AvgConf={track.get_avg_confidence():.2f}"
                    )
                    self.suspicious_timestamps.append({
                        "alert_number": self.alert_count,
                        "timestamp": timestamp,
                        "frame_id": self.frame_count,
                        "track_id": track.track_id,
                        "consecutive_frames": track.consecutive_frames,
                        "avg_confidence": track.get_avg_confidence(),
                    })

        return any_alert

    def _build_enriched_detections(self) -> List[Dict]:
        """
        Build enriched detection list with tracking metadata.

        Returns:
            List of detection dicts with track_id, persistent_frames, etc.
        """
        enriched = []
        for track in self.tracks:
            if not track.is_active or track.missing_frames > 0:
                continue
            det = track.to_dict()
            det["persistent_frames"] = track.consecutive_frames
            enriched.append(det)
        return enriched

    def get_best_detection(self) -> Optional[Dict]:
        """
        Get the highest-confidence active detection.

        Returns:
            Best detection dict, or None if no active detections
        """
        active = [t for t in self.tracks if t.is_active and t.missing_frames == 0]
        if not active:
            return None
        best = max(active, key=lambda t: t.get_avg_confidence())
        return best.to_dict()

    def get_status(self) -> Dict:
        """
        Get current tracker status summary.

        Returns:
            Status dictionary
        """
        active_tracks = [t for t in self.tracks if t.is_active and t.missing_frames == 0]
        best = self.get_best_detection()

        return {
            "phone_detected": len(active_tracks) > 0,
            "confidence": best["avg_confidence"] if best else 0.0,
            "tracked_objects": len(active_tracks),
            "alert_active": self.alert_active,
            "alert_count": self.alert_count,
            "frame_id": self.frame_count,
            "total_tracks": len(self.tracks),
        }

    def reset(self) -> None:
        """Reset tracker state for new video/session."""
        TrackedObject._id_counter = 0
        self.tracks = []
        self.frame_count = 0
        self.alert_active = False
        self.last_alert_time = 0.0
        self.alert_count = 0
        self.detection_history.clear()
        self.suspicious_timestamps = []
        logger.info("[✓] Tracker reset")
