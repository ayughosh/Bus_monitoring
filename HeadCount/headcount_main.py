# headcount_main.py
import sys
from bus_headcount_system import HeadCountSystem

# --- CONFIGURATION ---
# Adjust these settings for your specific setup.
CONFIG = {
    "camera_source": 2,  # 0 for integrated, 1 or 2 for external webcam
    "model_path": "models/yolov8n.pt",
    "confidence_threshold": 0.5  # How sure the model must be (0.0 to 1.0)
}


def main():
    """
    Initializes and runs the Bus Headcount System.
    """
    print("--- Starting Bus Headcount Monitoring System ---")

    head_counter = None
    try:
        head_counter = HeadCountSystem(
            camera_source=CONFIG["camera_source"],
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


if __name__ == '__main__':
    main()