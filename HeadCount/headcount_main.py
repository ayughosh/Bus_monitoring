# headcount_main.py
from bus_headcount_system import HeadCountSystem
import sys


def main():
    """
    Main function to run the head counting system.
    """
    print("[INFO] Starting Bus Headcount System...")

    # --- CONFIGURATION ---
    # Change this to the camera index for the entrance
    camera_id = 0

    try:
        # Initialize and run the system
        head_counter = HeadCountSystem(camera_source=camera_id)
        head_counter.run()
    except ConnectionError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[FATAL] An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

    print("[INFO] System shutdown.")


if __name__ == '__main__':
    main()