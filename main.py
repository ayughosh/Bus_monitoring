import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
import time
import os

class DrowsinessDetector:
    def __init__(self):
        # --- SETTINGS ---
        self.CAMERA_SOURCE = 0
        self.DLIB_LANDMARK_MODEL = "shape_predictor_68_face_landmarks.dat"
        self.RECALIBRATION_FRAMES = 20
        self.RECALIBRATION_INTERVAL_SECONDS = 60
        self.EAR_PERCENTAGE_THRESHOLD = 0.85
        self.EYE_CLOSURE_SECONDS_THRESHOLD = 2.0
        self.DROWSY_CONSEC_FRAMES = 25
        self.YAWN_MAR_THRESHOLD = 0.6
        self.YAWN_DURATION_SECONDS = 2.0
        self.SMILE_THRESHOLD = 0.8

        # --- STATE VARIABLES ---
        self.calibrated_ear_threshold = None
        self.is_calibrating = True
        self.calibration_ear_values = []
        self.last_recalibration_time = None
        self.eye_closure_start_time = None
        self.drowsy_counter = 0
        self.yawn_start_time = None

        # --- DLIB INITIALIZATION ---
        if not os.path.exists(self.DLIB_LANDMARK_MODEL):
            raise FileNotFoundError(f"[ERROR] Dlib model not found at '{self.DLIB_LANDMARK_MODEL}'")

        print("[INFO] Loading facial landmark predictor...")
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.DLIB_LANDMARK_MODEL)

        # Get landmark indexes
        self.lStart, self.lEnd = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        self.rStart, self.rEnd = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        self.mStart, self.mEnd = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    def mouth_aspect_ratio(self, mouth):
        A = dist.euclidean(mouth[2], mouth[10])
        B = dist.euclidean(mouth[4], mouth[8])
        C = dist.euclidean(mouth[0], mouth[6])
        return (A + B) / (2.0 * C)

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)

        status_text = "Active"
        alert_triggered = False

        if len(rects) > 0:
            largest_rect = max(rects, key=lambda rect: rect.width() * rect.height())
            shape = self.predictor(gray, largest_rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[self.lStart:self.lEnd]
            rightEye = shape[self.rStart:self.rEnd]
            mouth = shape[self.mStart:self.mEnd]

            ear = (self.eye_aspect_ratio(leftEye) + self.eye_aspect_ratio(rightEye)) / 2.0
            mar = self.mouth_aspect_ratio(mouth)

            # Draw contours for visualization
            cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (0, 255, 0), 1)

            if self.is_calibrating:
                status_text = "RECALIBRATING..."
                self.calibration_ear_values.append(ear)
                if len(self.calibration_ear_values) >= self.RECALIBRATION_FRAMES:
                    median_ear = np.median(self.calibration_ear_values)
                    self.calibrated_ear_threshold = median_ear * self.EAR_PERCENTAGE_THRESHOLD
                    self.is_calibrating = False
                    self.calibration_ear_values = []  # Reset for next time
                    print(
                        f"[{time.ctime()}] Recalibration complete. New EAR Threshold: {self.calibrated_ear_threshold:.2f}")
            else:
                smile_width = dist.euclidean(shape[48], shape[54])
                eye_distance = dist.euclidean(shape[36], shape[45])
                smile_ratio = smile_width / eye_distance
                is_smiling = smile_ratio > self.SMILE_THRESHOLD

                is_eyes_closed = ear < self.calibrated_ear_threshold
                is_mouth_open_wide = mar > self.YAWN_MAR_THRESHOLD

                if is_mouth_open_wide and not is_smiling:
                    if self.yawn_start_time is None:
                        self.yawn_start_time = time.time()
                    elif time.time() - self.yawn_start_time >= self.YAWN_DURATION_SECONDS:
                        alert_triggered = True
                else:
                    self.yawn_start_time = None

                if not alert_triggered and is_eyes_closed and not is_smiling:
                    if self.eye_closure_start_time is None:
                        self.eye_closure_start_time = time.time()
                    elif time.time() - self.eye_closure_start_time >= self.EYE_CLOSURE_SECONDS_THRESHOLD:
                        alert_triggered = True

                    self.drowsy_counter += 1
                    if self.drowsy_counter >= self.DROWSY_CONSEC_FRAMES:
                        alert_triggered = True
                else:
                    self.eye_closure_start_time = None
                    self.drowsy_counter = 0

                if alert_triggered:
                    status_text = "DROWSINESS ALERT!"
                elif is_smiling:
                    status_text = "Active (Smiling)"
        else:
            status_text = "No Face Detected"
            self.eye_closure_start_time = None
            self.drowsy_counter = 0
            self.yawn_start_time = None

        color = (0, 0, 255) if alert_triggered else (0, 255, 0)
        cv2.putText(frame, f"Status: {status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        return frame

    def run(self):
        print("[INFO] Starting video stream...")
        vs = cv2.VideoCapture(self.CAMERA_SOURCE)
        time.sleep(1.0)

        # Initial calibration
        self.is_calibrating = True
        self.last_recalibration_time = time.time()

        while True:
            # Periodic recalibration check
            if not self.is_calibrating and time.time() - self.last_recalibration_time > self.RECALIBRATION_INTERVAL_SECONDS:
                self.is_calibrating = True
                self.last_recalibration_time = time.time()

            ret, frame = vs.read()
            if not ret:
                break

            frame = cv2.resize(frame, (800, 600))

            processed_frame = self.process_frame(frame)

            if processed_frame is not None:
                cv2.imshow("Drowsiness Detector", processed_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        cv2.destroyAllWindows()
        vs.release()


if __name__ == "__main__":
    try:
        detector = DrowsinessDetector()
        detector.run()
    except Exception as e:
        print(e)