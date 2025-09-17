# bus_headcount_system.py
import cv2
from ultralytics import YOLO
import sys
import os


class HeadCountSystem:
    def __init__(self, camera_source=0, yolo_model_path="models/yolov8n.pt"):
        if not os.path.exists(yolo_model_path):
            raise FileNotFoundError(f"YOLO model not found at {yolo_model_path}.")

        self.cap = cv2.VideoCapture(camera_source)
        if not self.cap.isOpened():
            raise ConnectionError(f"Error: Could not open camera source {camera_source}")

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Zone is full-width and 25 pixels high
        zone_height = 25
        x_padding = 5
        self.COUNTING_ZONE = [
            x_padding,
            (self.frame_height - zone_height) // 2,
            self.frame_width - x_padding,
            (self.frame_height + zone_height) // 2
        ]

        print("[INFO] Loading YOLOv8 model from local file...")
        self.model = YOLO(yolo_model_path)
        print("[INFO] Model loaded successfully. Using built-in tracker.")

        self.track_data = {}
        self.counts = {'entry': 0, 'exit': 0}
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def _update_and_count(self, track_id, center_point):
        current_y = center_point[1]
        track_info = self.track_data.get(track_id, {'last_y': current_y, 'counted': False})
        last_y = track_info['last_y']

        zone_top_y = self.COUNTING_ZONE[1]
        zone_bottom_y = self.COUNTING_ZONE[3]

        crossed_in = (last_y < zone_top_y) and (current_y >= zone_top_y)
        if crossed_in and not track_info['counted']:
            self.counts['entry'] += 1
            track_info['counted'] = True

        crossed_out = (last_y > zone_bottom_y) and (current_y <= zone_bottom_y)
        if crossed_out and not track_info['counted']:
            self.counts['exit'] += 1
            track_info['counted'] = True

        is_inside_zone = self.COUNTING_ZONE[1] < current_y < self.COUNTING_ZONE[3]
        if not is_inside_zone:
            track_info['counted'] = False

        track_info['last_y'] = current_y
        self.track_data[track_id] = track_info

    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret: break

            results = self.model.track(frame, persist=True, classes=[0], verbose=False)

            zone_x1, zone_y1, zone_x2, zone_y2 = self.COUNTING_ZONE
            cv2.rectangle(frame, (zone_x1, zone_y1), (zone_x2, zone_y2), (0, 255, 255), 2)
            cv2.putText(frame, "Doorway Zone", (zone_x1 + 5, zone_y1 - 10), self.font, 0.6, (0, 255, 255), 2)

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)

                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), self.font, 0.6, (0, 255, 0), 2)
                    center_point = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    self._update_and_count(track_id, center_point)

            headcount = self.counts['entry'] - self.counts['exit']
            cv2.putText(frame, f"Entries: {self.counts['entry']}", (10, 40), self.font, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Exits: {self.counts['exit']}", (10, 80), self.font, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Onboard: {headcount}", (10, 120), self.font, 1, (255, 255, 0), 2)

            cv2.imshow("Bus Headcount System", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()