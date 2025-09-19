# HeadCount/headcount_main.py
"""
Main entry point for Bus Head Count System
Supports both the original HeadCountSystem and improved version with web interface
"""

import sys
import os
import threading
import time
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import the improved system first, fallback to original if needed
try:
    from bus_headcount_system import ImprovedHeadCountSystem, HeadCountWebServer

    IMPROVED_VERSION = True
    logger.info("Using improved head count system with web support")
except ImportError:
    try:
        from bus_headcount_system import HeadCountSystem

        IMPROVED_VERSION = False
        logger.warning("Using original head count system (no web interface)")
    except ImportError:
        logger.error("Could not import head count system")
        sys.exit(1)

# --- CONFIGURATION ---
CONFIG = {
    "camera_source": os.environ.get("CAMERA_INDEX", "2"),  # Read from env or default to 2
    "model_path": "models/yolov8n.onnx",
    "confidence_threshold": 0.5,
    "web_enabled": True,
    "web_host": "0.0.0.0",
    "web_port": 5000
}


def try_camera_sources():
    """
    Try different camera sources if the configured one fails
    Returns the first working camera source
    """
    import cv2

    # List of camera sources to try
    sources_to_try = [
        CONFIG["camera_source"],  # Try configured source first
        "0", "1", "2", "3",  # Try common indices
        "/dev/video0", "/dev/video1", "/dev/video2"  # Try device paths
    ]

    # Remove duplicates while preserving order
    seen = set()
    sources_to_try = [x for x in sources_to_try if not (x in seen or seen.add(x))]

    logger.info(f"Attempting to find working camera from: {sources_to_try}")

    for source in sources_to_try:
        try:
            # Convert to int if it's a digit string
            if isinstance(source, str) and source.isdigit():
                test_source = int(source)
            else:
                test_source = source

            # Try to open camera
            cap = cv2.VideoCapture(test_source)
            if cap.isOpened():
                # Test read
                ret, _ = cap.read()
                cap.release()
                if ret:
                    logger.info(f"✅ Camera found at source: {source}")
                    return test_source

        except Exception as e:
            logger.debug(f"Camera source {source} failed: {e}")
            continue

    return None


def run_improved_system(args):
    """
    Run the improved head count system with web interface
    """
    head_counter = None
    web_server = None

    try:
        # Try to find a working camera
        camera_source = try_camera_sources()
        if camera_source is None:
            logger.error("No working camera found. Please check:")
            logger.error("1. Camera is connected")
            logger.error("2. Camera permissions are set")
            logger.error("3. Camera is not in use by another program")
            sys.exit(1)

        # Initialize improved head count system
        logger.info(f"Initializing head count system with camera: {camera_source}")
        head_counter = ImprovedHeadCountSystem(
            camera_source=camera_source,
            yolo_model_path=CONFIG["model_path"],
            web_stream=args.web
        )

        if args.web:
            # Start web server in separate thread
            logger.info(f"Starting web server on {CONFIG['web_host']}:{CONFIG['web_port']}")
            web_server = HeadCountWebServer(
                head_counter,
                host=CONFIG['web_host'],
                port=CONFIG['web_port']
            )

            web_thread = threading.Thread(target=web_server.run)
            web_thread.daemon = True
            web_thread.start()

            # Give server time to start
            time.sleep(2)
            print("\n" + "=" * 50)
            print(f"✅ Web interface ready at:")
            print(f"   http://localhost:{CONFIG['web_port']}")
            print(f"   http://{CONFIG['web_host']}:{CONFIG['web_port']}")
            print("=" * 50 + "\n")

        # Run head counting
        head_counter.run()

    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        sys.exit(1)
    except ConnectionError as e:
        logger.error(f"Camera not accessible: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if head_counter:
            head_counter.cleanup()
        logger.info("System shutdown complete")


def run_original_system():
    """
    Run the original head count system (fallback)
    """
    print("--- Starting Bus Headcount Monitoring System (Original) ---")

    head_counter = None
    try:
        # Try to find a working camera
        camera_source = try_camera_sources()
        if camera_source is None:
            print("[FATAL ERROR] No working camera found", file=sys.stderr)
            sys.exit(1)

        head_counter = HeadCountSystem(
            camera_source=camera_source,
            yolo_model_path=CONFIG["model_path"]
        )
        head_counter.run(conf_threshold=CONFIG["confidence_threshold"])

    except FileNotFoundError as e:
        print(f"[FATAL ERROR] Model file not found. {e}", file=sys.stderr)
        sys.exit(1)
    except ConnectionError as e:
        print(f"[FATAL ERROR] Camera not accessible. {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[INFO] Program interrupted by user.")
    except Exception as e:
        print(f"[FATAL ERROR] An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if head_counter:
            head_counter.cleanup()
        print("--- System Shutdown ---")


def main():
    """
    Main entry point with argument parsing
    """
    parser = argparse.ArgumentParser(
        description='Bus Head Count Monitoring System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default camera and web interface
  python headcount_main.py --web

  # Run with specific camera
  python headcount_main.py --camera 1 --web

  # Run without web interface (console only)
  python headcount_main.py --no-web

  # Run with custom web port
  python headcount_main.py --web --port 8080
        """
    )

    parser.add_argument('--camera', type=str, default=CONFIG["camera_source"],
                        help='Camera source (index or device path)')
    parser.add_argument('--model', type=str, default=CONFIG["model_path"],
                        help='Path to YOLO model')
    parser.add_argument('--web', action='store_true', default=CONFIG["web_enabled"],
                        help='Enable web interface (default)')
    parser.add_argument('--no-web', dest='web', action='store_false',
                        help='Disable web interface')
    parser.add_argument('--host', type=str, default=CONFIG["web_host"],
                        help='Web server host (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=CONFIG["web_port"],
                        help='Web server port (default: 5000)')
    parser.add_argument('--conf', type=float, default=CONFIG["confidence_threshold"],
                        help='Detection confidence threshold (0.0-1.0)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')

    args = parser.parse_args()

    # Update configuration with arguments
    CONFIG["camera_source"] = args.camera
    CONFIG["model_path"] = args.model
    CONFIG["web_host"] = args.host
    CONFIG["web_port"] = args.port
    CONFIG["confidence_threshold"] = args.conf

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Print startup banner
    print("\n" + "=" * 60)
    print("🚌 BUS HEAD COUNT MONITORING SYSTEM")
    print("Karnataka Government Transport Safety Project")
    print("=" * 60)
    print(f"Camera Source: {CONFIG['camera_source']}")
    print(f"Model: {CONFIG['model_path']}")
    print(f"Web Interface: {'Enabled' if args.web else 'Disabled'}")
    if args.web:
        print(f"Web Server: http://{CONFIG['web_host']}:{CONFIG['web_port']}")
    print("=" * 60 + "\n")

    # Check which version to run
    if IMPROVED_VERSION:
        run_improved_system(args)
    else:
        logger.warning("Improved version not available, using original system")
        run_original_system()


if __name__ == '__main__':
    main()