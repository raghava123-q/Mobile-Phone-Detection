"""
==============================================================
Synthetic Test Video Generator - Mobile Phone Detection System
==============================================================
Generates test videos with simulated phone-like rectangles
for validating the detection pipeline without real cameras.

Creates scenarios:
  1. Phone appearing and disappearing
  2. Persistent phone presence (should trigger alert)
  3. Brief flash (should NOT trigger alert)
  4. Multiple objects
  5. Various confidence-simulating sizes/positions

Usage:
    python generate_test_video.py
==============================================================
"""

import cv2
import numpy as np
from pathlib import Path
from config import VIDEOS_DIR, OUTPUTS_DIR

# Video properties
WIDTH = 640
HEIGHT = 480
FPS = 30
DURATION_SEC = 15
TOTAL_FRAMES = FPS * DURATION_SEC


def create_background(frame_id: int) -> np.ndarray:
    """Create a realistic interview-like background."""
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    # Gradient background (office-like)
    for y in range(HEIGHT):
        ratio = y / HEIGHT
        r = int(60 + 30 * ratio)
        g = int(65 + 25 * ratio)
        b = int(75 + 20 * ratio)
        frame[y, :] = (b, g, r)

    # Simulated desk at bottom
    cv2.rectangle(frame, (0, HEIGHT - 80), (WIDTH, HEIGHT), (45, 40, 35), -1)

    # Simulated person silhouette (head + shoulders)
    center_x = WIDTH // 2
    head_y = 140

    # Head (oval)
    cv2.ellipse(frame, (center_x, head_y), (45, 55), 0, 0, 360, (120, 110, 100), -1)

    # Shoulders
    pts = np.array([
        [center_x - 120, 280],
        [center_x - 60, 200],
        [center_x, 190],
        [center_x + 60, 200],
        [center_x + 120, 280],
    ], np.int32)
    cv2.fillPoly(frame, [pts], (100, 95, 85))

    # Add slight noise for realism
    noise = np.random.randint(-5, 6, frame.shape, dtype=np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return frame


def draw_phone(frame: np.ndarray, x: int, y: int, w: int, h: int, alpha: float = 1.0):
    """Draw a realistic phone-like rectangle on the frame."""
    overlay = frame.copy()

    # Phone body (dark with rounded corners effect)
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (20, 20, 25), -1)

    # Phone screen (lighter interior)
    margin = 5
    cv2.rectangle(overlay, (x + margin, y + margin * 3),
                  (x + w - margin, y + h - margin * 2), (80, 100, 120), -1)

    # Screen content simulation (bright area)
    cv2.rectangle(overlay, (x + margin + 3, y + margin * 3 + 3),
                  (x + w - margin - 3, y + h - margin * 2 - 3), (140, 160, 200), -1)

    # Phone border highlight
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (50, 50, 55), 2)

    # Camera dot at top
    cv2.circle(overlay, (x + w // 2, y + 8), 3, (60, 60, 70), -1)

    # Blend with alpha
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    return frame


def draw_timestamp(frame: np.ndarray, frame_id: int, scenario: str):
    """Add timestamp and scenario info to frame."""
    time_sec = frame_id / FPS
    text = f"Frame: {frame_id} | Time: {time_sec:.1f}s | {scenario}"

    cv2.rectangle(frame, (0, 0), (WIDTH, 30), (0, 0, 0), -1)
    cv2.putText(frame, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return frame


def generate_test_video():
    """Generate the main test video with multiple scenarios."""
    output_path = str(VIDEOS_DIR / "test_interview_simulation.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, FPS, (WIDTH, HEIGHT))

    print("=" * 60)
    print("  Generating Synthetic Test Video")
    print(f"  Output: {output_path}")
    print(f"  Resolution: {WIDTH}x{HEIGHT} @ {FPS} FPS")
    print(f"  Duration: {DURATION_SEC}s ({TOTAL_FRAMES} frames)")
    print("=" * 60)

    for frame_id in range(TOTAL_FRAMES):
        frame = create_background(frame_id)
        scenario = ""

        # ── Scenario Timeline ───────────────────────────────────
        time_sec = frame_id / FPS

        # Scenario 1: No phone (0-3s) — baseline
        if time_sec < 3.0:
            scenario = "No phone (baseline)"

        # Scenario 2: Phone appears, persists (3-7s) — should trigger alert
        elif 3.0 <= time_sec < 7.0:
            scenario = "Phone persistent (SHOULD ALERT)"
            # Phone at right side, hand-held position
            px = 400 + int(5 * np.sin(frame_id * 0.1))
            py = 180 + int(3 * np.cos(frame_id * 0.08))
            frame = draw_phone(frame, px, py, 60, 110)

        # Scenario 3: Phone disappears briefly (7-8s)
        elif 7.0 <= time_sec < 8.0:
            scenario = "Phone hidden (cooldown)"

        # Scenario 4: Brief flash — 0.5s (8-8.5s) — should NOT alert
        elif 8.0 <= time_sec < 8.5:
            scenario = "Brief flash (should NOT alert)"
            frame = draw_phone(frame, 350, 200, 55, 100, alpha=0.7)

        # Scenario 5: No phone again (8.5-10s)
        elif 8.5 <= time_sec < 10.0:
            scenario = "No phone (recovery)"

        # Scenario 6: Phone at different position (10-13s) — alert
        elif 10.0 <= time_sec < 13.0:
            scenario = "Phone left side (SHOULD ALERT)"
            # Phone at left side
            px = 80 + int(8 * np.sin(frame_id * 0.05))
            py = 220 + int(5 * np.cos(frame_id * 0.06))
            frame = draw_phone(frame, px, py, 50, 95)

        # Scenario 7: Multiple rectangular objects (13-15s) — distractor test
        elif 13.0 <= time_sec < 15.0:
            scenario = "Distractor objects"
            # Distractor 1: Book/wallet (wrong aspect ratio)
            cv2.rectangle(frame, (100, 350), (250, 400), (40, 30, 20), -1)
            cv2.rectangle(frame, (100, 350), (250, 400), (80, 70, 50), 2)

            # Distractor 2: Remote (too thin)
            cv2.rectangle(frame, (500, 300), (530, 420), (30, 30, 35), -1)
            cv2.rectangle(frame, (500, 300), (530, 420), (60, 60, 65), 2)

        # Add timestamp overlay
        frame = draw_timestamp(frame, frame_id, scenario)
        writer.write(frame)

        if frame_id % (FPS * 2) == 0:
            print(f"  [{frame_id}/{TOTAL_FRAMES}] {scenario}")

    writer.release()
    print()
    print(f"  [✓] Test video saved: {output_path}")
    print(f"  [✓] Duration: {DURATION_SEC}s | Frames: {TOTAL_FRAMES}")
    print()
    return output_path


def generate_snapshot_images():
    """Generate individual test images for API testing."""
    print("Generating test snapshot images...")

    # Image 1: Clean frame (no phone)
    clean = create_background(0)
    clean_path = str(VIDEOS_DIR / "test_clean_frame.jpg")
    cv2.imwrite(clean_path, clean)
    print(f"  [✓] Clean frame: {clean_path}")

    # Image 2: Frame with phone
    with_phone = create_background(0)
    with_phone = draw_phone(with_phone, 380, 180, 60, 110)
    phone_path = str(VIDEOS_DIR / "test_phone_frame.jpg")
    cv2.imwrite(phone_path, with_phone)
    print(f"  [✓] Phone frame: {phone_path}")

    # Image 3: Multiple objects (distractor)
    distractors = create_background(0)
    cv2.rectangle(distractors, (100, 350), (250, 400), (40, 30, 20), -1)
    cv2.rectangle(distractors, (500, 300), (530, 420), (30, 30, 35), -1)
    dist_path = str(VIDEOS_DIR / "test_distractors.jpg")
    cv2.imwrite(dist_path, distractors)
    print(f"  [✓] Distractor frame: {dist_path}")

    print()


if __name__ == "__main__":
    generate_test_video()
    generate_snapshot_images()
    print("All test media generated. Ready for pipeline testing.")
