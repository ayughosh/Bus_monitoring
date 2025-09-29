import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
import time
import os
from collections import deque
from flask import Flask, Response, render_template_string, jsonify
import threading
import queue

app = Flask(__name__)
frame_queue = queue.Queue(maxsize=10)
current_status = {"status": "Active", "confidence": 0.0, "blink_rate": 0.0, "yawns": 0, "nods": 0}
detector_instance = None


class DrowsinessDetector:
    def __init__(self):
        # --- SETTINGS ---
        self.CAMERA_SOURCE = 10
        self.DLIB_LANDMARK_MODEL = "shape_predictor_68_face_landmarks.dat"

        # --- ADAPTIVE THRESHOLDS (Conservative for robustness) ---
        self.EAR_PERCENTILE = 25  # Use 25th percentile for threshold
        self.EAR_PERCENTAGE_THRESHOLD = 0.75  # More conservative than 0.85
        self.EYE_CLOSURE_SECONDS_THRESHOLD = 3.0  # Increased from 2.0
        self.DROWSY_CONSEC_FRAMES = 40  # Increased from 25
        self.YAWN_MAR_MULTIPLIER = 1.8  # Multiplier for baseline MAR
        self.YAWN_DURATION_SECONDS = 2.0  # Increased from 1.5
        self.SMILE_THRESHOLD = 0.8

        # --- SMOOTHING AND FILTERING ---
        self.SMOOTH_WINDOW = 5
        self.OUTLIER_FACTOR = 2.0
        self.MIN_FACE_SIZE = 80
        self.MAX_FACE_MOVEMENT = 0.6  # Max 30% size change

        # --- BLINK & FATIGUE DETECTION ---
        self.BLINK_CONSEC_FRAMES = 5  # Increased from 3
        self.FATIGUE_BLINK_RATE_SECONDS = 30
        self.FATIGUE_LOW_BLINK_THRESHOLD = 5  # Decreased from 8
        self.FATIGUE_HIGH_BLINK_THRESHOLD = 50  # Increased from 40
        self.FATIGUE_EVENT_WINDOW_SECONDS = 180  # Increased from 120
        self.FATIGUE_EVENT_THRESHOLD = 5  # Increased from 3

        # --- HEAD NOD DETECTION ---
        self.HEAD_NOD_FRAME_WINDOW = 30  # Increased from 20
        self.HEAD_NOD_THRESHOLD_PIXELS = 25  # Will be adaptive

        # --- RECALIBRATION SETTINGS ---
        self.CALIBRATION_FRAMES = 100  # Increased for better baseline
        self.RECALIBRATION_INTERVAL_SECONDS = 80  # Every 2 minutes
        self.FACE_TRACKING_TOLERANCE_PIXELS = 75
        self.RECALIBRATION_CANDIDATE_SECONDS = 3.0

        # --- CONFIDENCE SCORING ---
        self.MIN_CONFIDENCE = 0.65
        self.confidence_score = 0.0

        # --- STATE MANAGEMENT ---
        self.STATE = {"ACTIVE": 0, "FATIGUED": 1, "DROWSY": 2}
        self.current_state = self.STATE["ACTIVE"]
        self.state_transition_buffer = 10  # Frames to confirm state change
        self.state_counter = 0
        self.previous_state = self.STATE["ACTIVE"]

        # --- CALIBRATION VARIABLES ---
        self.calibrated_ear_threshold = None
        self.calibrated_mar_baseline = None
        self.calibrated_mar_yawn_threshold = None
        self.calibrated_face_size = None
        self.ear_std_dev = None
        self.mar_std_dev = None
        self.is_calibrating = False
        self.calibration_ear_values = []
        self.calibration_mar_values = []
        self.last_recalibration_time = None

        # --- DETECTION COUNTERS ---
        self.eye_closure_start_time = None
        self.drowsy_counter = 0
        self.yawn_start_time = None
        self.candidate_face_center = None
        self.candidate_start_time = None
        self.calibrated_face_centers = []
        self.yawn_count = 0
        self.nod_count = 0
        self.blink_counter = 0
        self.blink_frame_counter = 0
        self.display_blink_rate = 0
        self.blink_analysis_start_time = time.time()
        self.head_positions = deque(maxlen=self.HEAD_NOD_FRAME_WINDOW)
        self.is_yawning = False
        self.is_nodding = False
        self.fatigue_events = []

        # --- SMOOTHING BUFFERS ---
        self.ear_history = deque(maxlen=self.SMOOTH_WINDOW)
        self.mar_history = deque(maxlen=self.SMOOTH_WINDOW)
        self.last_face_size = None
        self.no_face_counter = 0

        # --- DLIB INITIALIZATION ---
        if not os.path.exists(self.DLIB_LANDMARK_MODEL):
            raise FileNotFoundError(f"[ERROR] Dlib model not found at '{self.DLIB_LANDMARK_MODEL}'")
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.DLIB_LANDMARK_MODEL)
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        (self.mStart, self.mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
        (self.nStart, self.nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]

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

    def remove_outliers(self, values):
        """Remove outliers using z-score method"""
        if len(values) < 3:
            return values
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            return values
        z_scores = [(x - mean) / std for x in values]
        filtered = [x for x, z in zip(values, z_scores) if abs(z) < self.OUTLIER_FACTOR]
        return filtered if len(filtered) > 0 else values

    def apply_smoothing(self, value, history):
        """Apply moving average smoothing"""
        history.append(value)
        return np.mean(history) if len(history) > 0 else value

    def calculate_confidence(self, ear, mar, face_size):
        """Calculate confidence score for detection"""
        confidence = 0.0

        # Eye closure confidence
        if self.calibrated_ear_threshold and self.ear_std_dev and self.ear_std_dev > 0:
            if ear < self.calibrated_ear_threshold:
                z_score = abs(self.calibrated_ear_threshold - ear) / self.ear_std_dev
                confidence += min(1.0, z_score / 3.0) * 0.4

        # Mouth state confidence
        if self.calibrated_mar_baseline and self.mar_std_dev and self.mar_std_dev > 0:
            mar_deviation = abs(mar - self.calibrated_mar_baseline) / self.mar_std_dev
            if mar_deviation > 2.0:
                confidence += min(1.0, mar_deviation / 4.0) * 0.3

        # Face stability confidence
        if self.calibrated_face_size and face_size:
            size_ratio = face_size / self.calibrated_face_size
            if 0.7 < size_ratio < 1.3:
                confidence += 0.2

        # Measurement consistency
        if len(self.ear_history) == self.SMOOTH_WINDOW:
            stability = 1.0 - min(1.0, np.std(self.ear_history) * 10)
            confidence += stability * 0.1

        return min(1.0, confidence)

    def recalibrate(self, vs):
        """Enhanced recalibration with robust statistics"""
        print(f"\n[{time.ctime()}] Starting enhanced recalibration...")
        print("[INFO] Please look naturally at camera with eyes open, mouth closed")

        self.is_calibrating = True
        self.calibration_ear_values = []
        self.calibration_mar_values = []
        face_sizes = []
        stable_frames = 0
        total_frames = 0

        while stable_frames < self.CALIBRATION_FRAMES and total_frames < self.CALIBRATION_FRAMES * 2:
            ret, frame = vs.read()
            if not ret:
                continue

            total_frames += 1
            frame = cv2.resize(frame, (800, 600))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = self.detector(gray, 0)

            progress = int((stable_frames / self.CALIBRATION_FRAMES) * 100)
            cv2.putText(frame, f"RECALIBRATING... {progress}%", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, "Keep eyes open, mouth closed, stay still", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            if len(rects) > 0:
                rect = rects[0]
                face_width = rect.width()

                if face_width < self.MIN_FACE_SIZE:
                    cv2.putText(frame, "Please move closer", (10, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    try:
                        while not frame_queue.empty():
                            frame_queue.get_nowait()
                        frame_queue.put(frame)
                    except:
                        pass
                    continue

                # Check for movement
                if self.last_face_size:
                    size_change = abs(face_width - self.last_face_size) / self.last_face_size
                    if size_change > 0.1:
                        cv2.putText(frame, "Please stay still", (10, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        self.last_face_size = face_width
                        try:
                            while not frame_queue.empty():
                                frame_queue.get_nowait()
                            frame_queue.put(frame)
                        except:
                            pass
                        continue

                self.last_face_size = face_width

                shape = self.predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                leftEye = shape[self.lStart:self.lEnd]
                rightEye = shape[self.rStart:self.rEnd]
                mouth = shape[self.mStart:self.mEnd]

                ear = (self.eye_aspect_ratio(leftEye) + self.eye_aspect_ratio(rightEye)) / 2.0
                mar = self.mouth_aspect_ratio(mouth)

                # Check stability
                if len(self.ear_history) > 0:
                    recent = list(self.ear_history)[-3:] if len(self.ear_history) >= 3 else self.ear_history
                    recent_avg = np.mean(recent)
                    if abs(ear - recent_avg) / recent_avg < 0.15:
                        self.calibration_ear_values.append(ear)
                        self.calibration_mar_values.append(mar)
                        face_sizes.append(face_width)
                        stable_frames += 1
                else:
                    self.ear_history.append(ear)

                # Visual feedback
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            try:
                while not frame_queue.empty():
                    frame_queue.get_nowait()
                frame_queue.put(frame)
            except:
                pass

        if len(self.calibration_ear_values) >= self.CALIBRATION_FRAMES * 0.8:
            # Remove outliers
            filtered_ears = self.remove_outliers(self.calibration_ear_values)
            filtered_mars = self.remove_outliers(self.calibration_mar_values)

            # Robust statistics
            ear_percentile = np.percentile(filtered_ears, self.EAR_PERCENTILE)
            self.calibrated_ear_threshold = ear_percentile * self.EAR_PERCENTAGE_THRESHOLD
            self.ear_std_dev = np.std(filtered_ears)

            self.calibrated_mar_baseline = np.median(filtered_mars)
            self.calibrated_mar_yawn_threshold = self.calibrated_mar_baseline * self.YAWN_MAR_MULTIPLIER
            self.mar_std_dev = np.std(filtered_mars)

            self.calibrated_face_size = np.median(face_sizes)

            # Adaptive head nod threshold
            self.HEAD_NOD_THRESHOLD_PIXELS = int(self.calibrated_face_size * 0.15)

            print(f"[{time.ctime()}] Recalibration successful!")
            print(f"  EAR Threshold: {self.calibrated_ear_threshold:.3f} (std: {self.ear_std_dev:.3f})")
            print(f"  MAR Baseline: {self.calibrated_mar_baseline:.3f}")
            print(f"  MAR Yawn Threshold: {self.calibrated_mar_yawn_threshold:.3f}")
            print(f"  Face Size: {self.calibrated_face_size:.0f}px\n")
        else:
            print("[WARNING] Recalibration failed: Insufficient stable samples")
            print("[INFO] Keeping previous calibration values")
            self.calibrated_ear_threshold = 0.20
            self.ear_std_dev = 0.04
            self.calibrated_mar_baseline = 0.30
            self.calibrated_mar_yawn_threshold = 0.65
            self.mar_std_dev = 0.10
            self.calibrated_face_size = 150.0  # A reasonable default face width
            self.HEAD_NOD_THRESHOLD_PIXELS = 22  # Based on default face size (150 * 0.15)

        self.is_calibrating = False
        self.last_recalibration_time = time.time()
        self.reset_counters()

        # Clear buffers
        self.ear_history.clear()
        self.mar_history.clear()

    def reset_counters(self):
        self.eye_closure_start_time = None
        self.drowsy_counter = 0
        self.yawn_start_time = None
        self.yawn_count = 0
        self.nod_count = 0
        self.blink_counter = 0
        self.fatigue_events = []
        self.state_counter = 0
        self.confidence_score = 0.0

    def adaptive_threshold(self, base_value, face_size):
        """Adjust thresholds based on face distance"""
        if not self.calibrated_face_size:
            return base_value
        size_ratio = face_size / self.calibrated_face_size
        # Adjust threshold based on distance
        adjusted = base_value * (2 - size_ratio) if size_ratio > 0.5 else base_value
        return np.clip(adjusted, base_value * 0.7, base_value * 1.3)

    def run(self):
        print("[INFO] Starting robust drowsiness detection system...")
        print("[INFO] Optimized for real-world bus environments\n")
        print("[INFO] Open browser at http://localhost:5000")

        vs = cv2.VideoCapture(self.CAMERA_SOURCE)
        time.sleep(2.0)  # Camera warm-up

        # Initial calibration
        self.recalibrate(vs)

        fps_counter = 0
        fps_start_time = time.time()
        fps = 0

        while True:
            # Auto-recalibration check
            if not self.is_calibrating and time.time() - self.last_recalibration_time > self.RECALIBRATION_INTERVAL_SECONDS:
                print("\n[INFO] Periodic recalibration triggered")
                self.recalibrate(vs)

            ret, frame = vs.read()
            if not ret:
                break

            frame = cv2.resize(frame, (800, 600))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # FPS calculation
            fps_counter += 1
            if fps_counter % 30 == 0:
                fps = 30 / (time.time() - fps_start_time)
                fps_start_time = time.time()

            rects = self.detector(gray, 0)

            status_text = "No Face Detected"
            color = (128, 128, 128)

            if len(rects) == 0:
                self.no_face_counter += 1
                if self.no_face_counter > 30:
                    self.eye_closure_start_time = None
                    self.yawn_start_time = None
                    self.confidence_score *= 0.95
            else:
                self.no_face_counter = 0

                # Use largest face
                largest_rect = max(rects, key=lambda rect: rect.width() * rect.height())
                face_width = largest_rect.width()

                if face_width < self.MIN_FACE_SIZE:
                    status_text = "Face too far"
                    color = (0, 165, 255)
                    cv2.putText(frame, f"Status: {status_text}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    try:
                        # Clear old frames
                        while not frame_queue.empty():
                            frame_queue.get_nowait()
                        # Add new frame
                        frame_queue.put(frame)
                    except:
                        pass
                    continue

                # Check for sudden movement
                if self.last_face_size:
                    size_change = abs(face_width - self.last_face_size) / self.last_face_size
                    if size_change > self.MAX_FACE_MOVEMENT:
                        self.last_face_size = face_width
                        continue

                self.last_face_size = face_width

                (x, y, w, h) = (largest_rect.left(), largest_rect.top(),
                                largest_rect.width(), largest_rect.height())
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

                if self.is_calibrating:
                    status_text = "Calibrating..."
                    color = (255, 255, 0)
                else:
                    shape = self.predictor(gray, largest_rect)
                    shape = face_utils.shape_to_np(shape)

                    leftEye = shape[self.lStart:self.lEnd]
                    rightEye = shape[self.rStart:self.rEnd]
                    mouth = shape[self.mStart:self.mEnd]
                    nose = shape[self.nStart:self.nEnd]

                    # Calculate raw metrics
                    ear_raw = (self.eye_aspect_ratio(leftEye) + self.eye_aspect_ratio(rightEye)) / 2.0
                    mar_raw = self.mouth_aspect_ratio(mouth)

                    # Apply smoothing
                    ear = self.apply_smoothing(ear_raw, self.ear_history)
                    mar = self.apply_smoothing(mar_raw, self.mar_history)

                    # Calculate confidence
                    self.confidence_score = self.calculate_confidence(ear, mar, face_width)

                    # Adaptive thresholds
                    current_ear_threshold = self.adaptive_threshold(self.calibrated_ear_threshold, face_width)
                    current_mar_threshold = self.adaptive_threshold(self.calibrated_mar_yawn_threshold, face_width)

                    is_eyes_closed = ear < current_ear_threshold
                    is_mouth_open_wide = mar > current_mar_threshold

                    # --- ROBUST FATIGUE ANALYSIS ---
                    if is_eyes_closed:
                        self.blink_frame_counter += 1
                    else:
                        if self.blink_frame_counter >= self.BLINK_CONSEC_FRAMES:
                            self.blink_counter += 1
                        self.blink_frame_counter = 0

                    fatigue_detected = False
                    if time.time() - self.blink_analysis_start_time > self.FATIGUE_BLINK_RATE_SECONDS:
                        self.display_blink_rate = (self.blink_counter / self.FATIGUE_BLINK_RATE_SECONDS) * 60
                        if (self.display_blink_rate < self.FATIGUE_LOW_BLINK_THRESHOLD or
                                self.display_blink_rate > self.FATIGUE_HIGH_BLINK_THRESHOLD):
                            if self.confidence_score > 0.5:  # Require confidence for fatigue
                                fatigue_detected = True
                                self.fatigue_events.append(time.time())
                        self.blink_counter = 0
                        self.blink_analysis_start_time = time.time()

                    # Check fatigue event history
                    self.fatigue_events = [event for event in self.fatigue_events
                                           if time.time() - event < self.FATIGUE_EVENT_WINDOW_SECONDS]
                    if len(self.fatigue_events) >= self.FATIGUE_EVENT_THRESHOLD:
                        fatigue_detected = True

                    # --- ROBUST DROWSINESS DETECTION ---
                    drowsiness_event = False

                    # Yawn detection with confidence
                    if is_mouth_open_wide and self.confidence_score > 0.5:
                        if self.yawn_start_time is None:
                            self.yawn_start_time = time.time()
                        elif time.time() - self.yawn_start_time >= self.YAWN_DURATION_SECONDS:
                            if not self.is_yawning:
                                self.yawn_count += 1
                                self.fatigue_events.append(time.time())
                                self.is_yawning = True
                            drowsiness_event = True
                    else:
                        self.yawn_start_time = None
                        self.is_yawning = False

                    # Head nod detection with smoothing
                    if not is_mouth_open_wide:
                        nose_tip_y = np.mean([p[1] for p in nose])
                        self.head_positions.append(nose_tip_y)
                        if len(self.head_positions) >= self.HEAD_NOD_FRAME_WINDOW:
                            positions = np.array(self.head_positions)
                            smoothed = np.convolve(positions, np.ones(5) / 5, mode='valid')
                            if len(smoothed) > 0:
                                adaptive_nod_threshold = self.adaptive_threshold(
                                    self.HEAD_NOD_THRESHOLD_PIXELS, face_width
                                )
                                if max(smoothed) - smoothed[-1] > adaptive_nod_threshold:
                                    if not self.is_nodding:
                                        self.nod_count += 1
                                        self.fatigue_events.append(time.time())
                                        self.is_nodding = True
                                    drowsiness_event = True
                                else:
                                    self.is_nodding = False

                    # Eye closure detection with confidence
                    if is_eyes_closed and self.confidence_score > self.MIN_CONFIDENCE:
                        if self.eye_closure_start_time is None:
                            self.eye_closure_start_time = time.time()
                        elif time.time() - self.eye_closure_start_time >= self.EYE_CLOSURE_SECONDS_THRESHOLD:
                            drowsiness_event = True
                    else:
                        self.eye_closure_start_time = None

                    # --- ROBUST STATE MACHINE WITH HYSTERESIS ---
                    target_state = self.current_state

                    if self.current_state == self.STATE["ACTIVE"]:
                        if fatigue_detected and self.confidence_score > 0.5:
                            target_state = self.STATE["FATIGUED"]
                    elif self.current_state == self.STATE["FATIGUED"]:
                        if drowsiness_event and self.confidence_score > self.MIN_CONFIDENCE:
                            target_state = self.STATE["DROWSY"]
                        elif not fatigue_detected and self.confidence_score < 0.3:
                            target_state = self.STATE["ACTIVE"]
                    elif self.current_state == self.STATE["DROWSY"]:
                        if not drowsiness_event and self.confidence_score < 0.5:
                            target_state = self.STATE["FATIGUED"]

                    # Apply hysteresis
                    if target_state != self.current_state:
                        if target_state == self.previous_state:
                            self.state_counter += 1
                            if self.state_counter >= self.state_transition_buffer:
                                self.current_state = target_state
                                self.state_counter = 0
                        else:
                            self.previous_state = target_state
                            self.state_counter = 1
                    else:
                        self.state_counter = 0

                    # Determine status based on state
                    if self.current_state == self.STATE["DROWSY"]:
                        status_text = "DROWSINESS ALERT!"
                        color = (0, 0, 255)
                    elif self.current_state == self.STATE["FATIGUED"]:
                        status_text = "Fatigue Detected"
                        color = (0, 165, 255)
                    else:
                        status_text = "Active"
                        color = (0, 255, 0)

                    # Draw eye contours
                    leftEyeHull = cv2.convexHull(leftEye)
                    rightEyeHull = cv2.convexHull(rightEye)
                    cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                    cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            # Display information
            cv2.putText(frame, f"Status: {status_text}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame, f"Confidence: {self.confidence_score:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            time_to_decide = self.FATIGUE_BLINK_RATE_SECONDS - (time.time() - self.blink_analysis_start_time)
            info_text_1 = f"Blink Rate: {self.display_blink_rate:.1f} bpm"
            info_text_2 = f"Fatigue Check In: {int(time_to_decide)}s"
            info_text_3 = f"Yawns: {self.yawn_count} | Nods: {self.nod_count}"

            cv2.putText(frame, info_text_1, (550, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, info_text_2, (550, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, info_text_3, (550, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Instructions
            cv2.putText(frame, "Press 'q' to quit | 'r' to recalibrate", (10, 580),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            current_status = {
                "status": status_text if 'status_text' in locals() else "No Face Detected",
                "confidence": float(self.confidence_score),
                "blink_rate": float(self.display_blink_rate),
                "yawns": int(self.yawn_count),
                "nods": int(self.nod_count)
            }
            try:
                # Clear old frames
                while not frame_queue.empty():
                    frame_queue.get_nowait()
                # Add new frame
                frame_queue.put(frame)
            except:
                pass

        vs.release()


# Add Flask routes
@app.route('/')
def index():
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>Bus Driver Drowsiness Detection</title>
    <style>
        body {
            font-family: Arial;
            margin: 0;
            padding: 20px;
            background: #1a1a1a;
            color: white;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            color: #00ff00;
        }
        .video-container {
            text-align: center;
            margin: 20px 0;
            border: 3px solid #333;
            border-radius: 10px;
            overflow: hidden;
        }
        .video-container img {
            max-width: 100%;
            height: auto;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .stat-box {
            background: #2a2a2a;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
        .status-active { color: #00ff00; }
        .status-warning { color: #ffa500; }
        .status-alert { color: #ff0000; }
        button {
            background: #00ff00;
            color: #000;
            border: none;
            padding: 10px 20px;
            margin: 5px;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
        }
        .controls {
            text-align: center;
            margin: 20px 0;
        }
    </style>
    <script>
        function updateStatus() {
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('status').textContent = data.status;
                    document.getElementById('status').className = 
                        data.status.includes('ALERT') ? 'status-alert' :
                        data.status.includes('Fatigue') ? 'status-warning' : 'status-active';
                    document.getElementById('confidence').textContent = data.confidence.toFixed(2);
                    document.getElementById('blink_rate').textContent = data.blink_rate.toFixed(1) + ' bpm';
                    document.getElementById('yawns').textContent = data.yawns;
                    document.getElementById('nods').textContent = data.nods;
                });
        }

        function recalibrate() {
            if(confirm('Start recalibration?')) {
                alert('Please refresh the page to recalibrate');
            }
        }

        setInterval(updateStatus, 500);
    </script>
</head>
<body>
    <div class="container">
        <h1>🚌 Bus Driver Drowsiness Detection</h1>

        <div class="stats">
            <div class="stat-box">
                <h3>Status</h3>
                <h2 id="status" class="status-active">Active</h2>
            </div>
            <div class="stat-box">
                <h3>Confidence</h3>
                <h2 id="confidence">0.00</h2>
            </div>
            <div class="stat-box">
                <h3>Blink Rate</h3>
                <h2 id="blink_rate">0 bpm</h2>
            </div>
            <div class="stat-box">
                <h3>Yawns</h3>
                <h2 id="yawns">0</h2>
            </div>
            <div class="stat-box">
                <h3>Nods</h3>
                <h2 id="nods">0</h2>
            </div>
        </div>

        <div class="video-container">
            <img src="/video_feed" />
        </div>

        <div class="controls">
            <button onclick="recalibrate()">Recalibrate</button>
        </div>
    </div>
</body>
</html>
    ''')


@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            try:
                frame = frame_queue.get(timeout=1)
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            except:
                continue

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status')
def status():
    return jsonify(current_status)


# Modify the main execution block:
if __name__ == "__main__":
    try:
        # Create detector instance
        detector_instance = DrowsinessDetector()

        # Start detection in separate thread
        detection_thread = threading.Thread(target=detector_instance.run)
        detection_thread.daemon = True
        detection_thread.start()

        # Start Flask web server
        print("\n[INFO] Starting web interface...")
        print("[INFO] Open browser at http://localhost:5000")
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

    except Exception as e:
        print(f"[FATAL ERROR] {e}")
        import traceback

        traceback.print_exc()