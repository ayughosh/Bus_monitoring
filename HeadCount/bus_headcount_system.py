# bus_headcount_system.py
import cv2
from ultralytics import YOLO
import os
import math


class HeadCountSystem:
    """
    A robust system for detecting and counting people's heads as they cross
    a predefined zone in a video feed.
    """

    def __init__(self, camera_source=0, yolo_model_path="models/yolov8n.pt"):
        """
        Initializes the HeadCountSystem.
        """
        if not os.path.exists(yolo_model_path):
            raise FileNotFoundError(f"YOLO model not found at '{yolo_model_path}'.")

        self.cap = cv2.VideoCapture(camera_source)
        if not self.cap.isOpened():
            raise ConnectionError(f"Could not open camera source '{camera_source}'.")

        # --- Frame and Zone Configuration ---
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        zone_height = 25
        self.COUNTING_ZONE = [
            5, (self.frame_height - zone_height) // 2,
               self.frame_width - 5, (self.frame_height + zone_height) // 2
        ]
        self.zone_center_y = (self.COUNTING_ZONE[1] + self.COUNTING_ZONE[3]) / 2

        # --- Model Initialization ---
        print(f"[INFO] Loading model '{yolo_model_path}'...")
        self.model = YOLO(yolo_model_path)
        print("[INFO] Model loaded successfully.")

        # --- System State ---
        self.track_data = {}
        self.counts = {'entry': 0, 'exit': 0}
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def _update_and_count(self, track_id, center_point):
        """
        Updates the state of a tracked person and counts them if they cross
        into the zone. This is the core counting logic.
        """
        x, y = center_point
        track_info = self.track_data.get(track_id, {'state': 'uncounted', 'path': []})
        track_info['path'].append(center_point)
        if len(track_info['path']) > 10: track_info['path'].pop(0)

        zone_x1, zone_y1, zone_x2, zone_y2 = self.COUNTING_ZONE
        is_inside_zone = zone_x1 < x < zone_x2 and zone_y1 < y < zone_y2

        # --- MODIFIED: Simplified and more responsive reset logic ---
        if track_info['state'] == 'uncounted' and is_inside_zone:
            # Person has just entered the zone, let's count them
            if len(track_info['path']) > 1:
                last_y = track_info['path'][-2][1]
                if last_y < self.zone_center_y:
                    self.counts['entry'] += 1
                    track_info['state'] = 'counted'  # Mark as counted
                else:
                    self.counts['exit'] += 1
                    track_info['state'] = 'counted'  # Mark as counted
        elif not is_inside_zone and track_info['state'] == 'counted':
            # Person has left the zone, reset their state to be countable again
            track_info['state'] = 'uncounted'

        self.track_data[track_id] = track_info

    def _visualize(self, frame, tracks):
        """Draws all visualizations on the frame."""
        # Draw the counting zone
        zone_x1, zone_y1, zone_x2, zone_y2 = self.COUNTING_ZONE
        cv2.rectangle(frame, (zone_x1, zone_y1), (zone_x2, zone_y2), (0, 255, 255), 2)
        cv2.putText(frame, "Doorway Zone", (zone_x1 + 5, zone_y1 - 10), self.font, 0.6, (0, 255, 255), 2)

        # Draw bounding boxes and tracking points for each tracked person
        for track in tracks:
            head_box, head_center, track_id = track
            cv2.rectangle(frame, (head_box[0], head_box[1]), (head_box[2], head_box[3]), (0, 255, 0), 2)
            cv2.circle(frame, head_center, 5, (255, 0, 0), -1)
            cv2.putText(frame, f"ID: {track_id}", (head_center[0] - 15, head_center[1] - 15), self.font, 0.6,
                        (0, 255, 0), 2)

        # Draw the counts
        headcount = self.counts['entry'] - self.counts['exit']
        cv2.putText(frame, f"Entries: {self.counts['entry']}", (20, 40), self.font, 1.2, (0, 255, 0), 3)
        cv2.putText(frame, f"Exits: {self.counts['exit']}", (20, 85), self.font, 1.2, (0, 0, 255), 3)
        cv2.putText(frame, f"Onboard: {headcount}", (20, 130), self.font, 1.2, (255, 255, 0), 3)

    def run(self, conf_threshold=0.5):
        """Starts the main video processing loop."""
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("[WARNING] End of video stream or camera disconnected.")
                break

            results = self.model.track(frame, persist=True, classes=[0], conf=conf_threshold, verbose=False)

            processed_tracks = []
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)

                for box, track_id in zip(boxes, track_ids):
                    # --- MODIFIED: Centralized and corrected head estimation ---
                    x1, y1, x2, y2 = box
                    head_y2 = y1 + int((y2 - y1) * 0.30)
                    head_box = (x1, y1, x2, head_y2)
                    head_center = (int((x1 + x2) / 2), int((y1 + head_y2) / 2))

                    self._update_and_count(track_id, head_center)
                    processed_tracks.append((head_box, head_center, track_id))

            self._visualize(frame, processed_tracks)

            cv2.imshow("Bus Headcount System", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def cleanup(self):
        """Releases video capture and destroys all OpenCV windows."""
        print("[INFO] Releasing resources.")
        self.cap.release()
        cv2.destroyAllWindows()