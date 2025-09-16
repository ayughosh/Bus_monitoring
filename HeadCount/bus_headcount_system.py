# bus_headcount_system.py
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import sys
import os  # Import the os module


class HeadCountSystem:
    # --- MODIFICATION: Point directly to the local model file ---
    def __init__(self, camera_source=0, yolo_model_path="models/yolov8n.pt"):
        # --- Check if the model file exists before proceeding ---
        if not os.path.exists(yolo_model_path):
            raise FileNotFoundError(f"YOLO model not found at {yolo_model_path}. Please download it.")

        self.cap = cv2.VideoCapture(camera_source)
        if not self.cap.isOpened():
            raise ConnectionError(f"Error: Could not open camera source {camera_source}")

        print("[INFO] Loading YOLOv8 model from local file...")
        self.model = YOLO(yolo_model_path)
        print("[INFO] Model loaded successfully.")

        self.tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)
        self.line_coords = None
        self.track_history = {}
        self.counts = {'entry': 0, 'exit': 0}
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.line_coords is None:
                self.line_coords = [(x, y)]
            elif len(self.line_coords) == 1:
                self.line_coords.append((x, y))

    def calibrate(self):
        print("\n--- CALIBRATION MODE ---")
        print("Camera starting... Click two points to define the line. Press 'c' to confirm.")
        cv2.namedWindow("Calibration")
        cv2.setMouseCallback("Calibration", self._mouse_callback)
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[ERROR] Failed to grab frame from webcam. Exiting.", file=sys.stderr)
                break

            if self.line_coords:
                if len(self.line_coords) == 1:
                    cv2.circle(frame, self.line_coords[0], 5, (0, 0, 255), -1)
                elif len(self.line_coords) == 2:
                    cv2.line(frame, self.line_coords[0], self.line_coords[1], (0, 255, 0), 2)

            cv2.imshow("Calibration", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c') and self.line_coords and len(self.line_coords) == 2:
                break
            elif key == ord('q'):
                self.cap.release()
                cv2.destroyAllWindows()
                sys.exit(0)
        cv2.destroyWindow("Calibration")

    def run(self):
        self.calibrate()
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret: break

            results = self.model(frame, classes=[0], verbose=False)
            detections = [([int(r.xyxy[0][0]), int(r.xyxy[0][1]), int(r.xyxy[0][2] - r.xyxy[0][0]),
                            int(r.xyxy[0][3] - r.xyxy[0][1])], float(r.conf[0]), 0) for r in results[0].boxes]
            tracks = self.tracker.update_tracks(detections, frame=frame)

            if self.line_coords:
                cv2.line(frame, self.line_coords[0], self.line_coords[1], (0, 255, 0), 2)

            for track in tracks:
                if not track.is_confirmed() or track.time_since_update > 1: continue
                # ... [Rest of the run method is the same] ...

            cv2.imshow("Bus Headcount System", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        self.cap.release()
        cv2.destroyAllWindows()

    # The _check_line_crossing method remains the same
    def _check_line_crossing(self, track_id, center_point):
        if not self.line_coords or len(self.line_coords) < 2: return
        if track_id not in self.track_history:
            self.track_history[track_id] = {'history': [], 'crossed': False}
        history = self.track_history[track_id]['history']
        history.append(center_point)
        if len(history) > 2: history.pop(0)
        if len(history) == 2 and not self.track_history[track_id]['crossed']:
            p1, p2 = self.line_coords
            prev_point, curr_point = history
            line_eq = lambda p: (p2[1] - p1[1]) * p[0] + (p1[0] - p2[0]) * p[1] + (p2[0] * p1[1] - p1[0] * p2[1])
            if line_eq(prev_point) * line_eq(curr_point) < 0:
                self.track_history[track_id]['crossed'] = True
                if curr_point[1] > prev_point[1]:
                    self.counts['entry'] += 1
                else:
                    self.counts['exit'] += 1
        elif self.track_history[track_id]['crossed']:
            dist_to_line = abs(cv2.pointPolygonTest(np.array(self.line_coords), center_point, True))
            if dist_to_line > 50: self.track_history[track_id]['crossed'] = False