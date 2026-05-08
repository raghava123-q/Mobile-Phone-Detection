"""
==============================================================
Main Application Entry Point - Mobile Phone Detection System
==============================================================
Unified entry point that supports three modes:
  1. Flask API server  (default / --mode api)
  2. Webcam live demo (--mode webcam)
  3. Video file processing (--mode video --source path/to/video.mp4)

Usage examples:
  python app.py                                   # Start API server
  python app.py --mode webcam                     # Live webcam detection
  python app.py --mode video --source video.mp4   # Process video file
  python app.py --mode api --host 0.0.0.0 --port 5000
==============================================================
"""

import argparse
import sys

from utils.logger import setup_logger

logger = setup_logger("App")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mobile Phone Detection System - YOLOv8",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py                                     # Start Flask API
  python app.py --mode webcam                       # Live webcam
  python app.py --mode webcam --camera 0 --save     # Save webcam output
  python app.py --mode video --source video.mp4     # Process video
  python app.py --mode api --port 8080              # API on port 8080
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["api", "webcam", "video"],
        default="api",
        help="Run mode: api (Flask server), webcam (live), video (file) [default: api]",
    )

    # Video source
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to input video file (required for --mode video)",
    )

    # Webcam
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam device ID [default: 0]",
    )

    # API server
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="API server host [default: 0.0.0.0]",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="API server port [default: 5000]",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Flask debug mode",
    )

    # Detection settings
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="Confidence threshold [default: from config.py]",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="YOLOv8 model path or name [default: yolov8n.pt]",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Compute device [default: auto]",
    )

    # Output
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save annotated output video",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable live display window (headless mode)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output video path (optional)",
    )

    return parser.parse_args()


def run_api_mode(args):
    """Launch the Flask API server."""
    logger.info("=" * 60)
    logger.info("  Starting Flask API Server")
    logger.info(f"  Host: {args.host}:{args.port}")
    logger.info(f"  Debug: {args.debug}")
    logger.info("=" * 60)

    from api.routes import app
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        use_reloader=False,
    )


def run_webcam_mode(args):
    """Run live webcam detection."""
    logger.info("=" * 60)
    logger.info("  Starting Webcam Detection")
    logger.info(f"  Camera ID: {args.camera}")
    logger.info("  Controls: q=quit | s=snapshot | r=reset tracker")
    logger.info("=" * 60)

    from inference import InferencePipeline

    pipeline = InferencePipeline(
        model_path=args.model,
        confidence_threshold=args.confidence,
        device=args.device,
        show_display=not args.no_display,
        save_output=args.save,
    )

    try:
        summary = pipeline.process_webcam(
            camera_id=args.camera,
            output_path=args.output,
        )
        logger.info("Webcam session summary:")
        logger.info(f"  Total frames:    {summary['total_frames']}")
        logger.info(f"  Time:            {summary['total_time_sec']:.1f}s")
        logger.info(f"  Phone detected:  {summary['phone_detected_frames']} frames")
        logger.info(f"  Total alerts:    {summary['total_alerts']}")
    except RuntimeError as e:
        logger.error(f"Webcam error: {e}")
        sys.exit(1)


def run_video_mode(args):
    """Process a video file."""
    if not args.source:
        logger.error("--source is required for --mode video")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("  Processing Video File")
    logger.info(f"  Source: {args.source}")
    logger.info("=" * 60)

    from inference import InferencePipeline

    pipeline = InferencePipeline(
        model_path=args.model,
        confidence_threshold=args.confidence,
        device=args.device,
        show_display=not args.no_display,
        save_output=args.save or True,  # Always save for video mode
    )

    try:
        summary = pipeline.process_video(
            video_path=args.source,
            output_path=args.output,
        )
        logger.info("Video processing summary:")
        for key, value in summary.items():
            if key not in ("suspicious_timestamps", "annotated_frame"):
                logger.info(f"  {key}: {value}")
    except (FileNotFoundError, RuntimeError) as e:
        logger.error(f"Video error: {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    print()
    print("=" * 60)
    print("  Mobile Phone Detection System")
    print("  YOLOv8 + OpenCV + Flask | Interview Proctoring")
    print("=" * 60)
    print()

    args = parse_args()

    if args.mode == "api":
        run_api_mode(args)
    elif args.mode == "webcam":
        run_webcam_mode(args)
    elif args.mode == "video":
        run_video_mode(args)


if __name__ == "__main__":
    main()
