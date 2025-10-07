import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
import time
import os
from collections import deque
from flask import Flask, Response, render_template_string, jsonify, request
import threading
import queue
import json

app = Flask(__name__)
frame_queue = queue.Queue(maxsize=2)  # Reduced queue size for lower latency
current_status = {"status": "Active", "confidence": 0.0, "blink_rate": 0.0, "yawns": 0, "nods": 0, "fatigue_check": 0, "fps": 0.0}
detector_instance = None
recalibrate_flag = False


class SimpleDrowsinessDetector:
    def __init__(self):
        # --- CAMERA SETTINGS ---
        # Set to 0 for webcam or RTSP URL for IP camera
        # Example RTSP URLs:
        # - Hikvision: "rtsp://username:password@ip_address:554/Streaming/Channels/101"
        # - Generic: "rtsp://username:password@ip_address:port/stream"
        self.RTSP_URL = "rtsp://admin:admin@192.168.1.64:554/Streaming/Channels/101"  # Change this to your camera's RTSP URL
        self.USE_RTSP = True  # Set to False to use webcam (0), True to use RTSP
        self.CAMERA_SOURCE = self.RTSP_URL if self.USE_RTSP else 0
        self.DLIB_LANDMARK_MODEL = "shape_predictor_68_face_landmarks.dat"
        self.CONFIG_FILE = "drowsiness_config.json"

        # Load config or use defaults
        self.load_config()

    def load_config(self):
        """Load configuration from JSON file or use defaults"""
        defaults = {
            "EAR_PERCENTILE": 25,
            "EAR_PERCENTAGE_THRESHOLD": 0.75,
            "EYE_CLOSURE_SECONDS_THRESHOLD": 4.0,
            "YAWN_MAR_MULTIPLIER": 1.4,
            "YAWN_DURATION_SECONDS": 0.8,
            "YAWN_MIN_FRAMES": 8,
            "YAWN_MIN_DURATION": 0.8,
            "YAWN_MAX_DURATION": 6.0,
            "YAWN_OPENING_RATE_THRESH": 0.02,
            "YAWN_MAR_INCREASE_THRESH": 1.35,
            "YAWN_COOLDOWN_FRAMES": 20,
            "BLINK_CONSEC_FRAMES": 4,
            "FATIGUE_BLINK_RATE_SECONDS": 90,
            "FATIGUE_LOW_BLINK_THRESHOLD": 1,
            "FATIGUE_HIGH_BLINK_THRESHOLD": 80,
            "DROWSY_MIN_YAWNS": 5,
            "MIN_CONFIDENCE": 0.35,
            "FATIGUE_EVENT_THRESHOLD": 8,
            "DROWSY_MIN_FATIGUE_EVENTS": 12,
            "AUTO_RECALIBRATE_SECONDS": 60,
            "FACE_LOST_RECALIBRATE_SECONDS": 3.0,
            "FACE_ACTIVE_THRESHOLD_SECONDS": 2.0
        }

        if os.path.exists(self.CONFIG_FILE):
            try:
                with open(self.CONFIG_FILE, 'r') as f:
                    loaded_config = json.load(f)
                    defaults.update(loaded_config)
                print(f"[INFO] Loaded config from {self.CONFIG_FILE}")
            except Exception as e:
                print(f"[WARNING] Could not load config: {e}. Using defaults.")

        # Apply configuration
        for key, value in defaults.items():
            setattr(self, key, value)

        # Non-configurable settings
        self.SMOOTH_WINDOW = 7
        self.OUTLIER_FACTOR = 2.5
        self.MIN_FACE_SIZE = 60
        self.HEAD_NOD_FRAME_WINDOW = 60
        self.HEAD_NOD_THRESHOLD_MULTIPLIER = 0.25
        self.NOD_SMOOTHING_KERNEL = 15
        self.STATE = {"ACTIVE": 0, "FATIGUED": 1, "DROWSY": 2}
        self.current_state = self.STATE["ACTIVE"]
        self.state_transition_buffer = 30
        self.state_counter = 0
        self.previous_state = self.STATE["ACTIVE"]
        self.DROWSY_MIN_FATIGUE_EVENTS = 12
        self.DROWSY_MIN_EYE_CLOSURE_DURATION = 6.0
        self.CALIBRATION_FRAMES = 60
        self.RECALIBRATION_INTERVAL_SECONDS = 300
        self.confidence_score = 0.0
        self.FATIGUE_EVENT_WINDOW_SECONDS = 420
        self.FATIGUE_EVENT_THRESHOLD = 8

        # --- CALIBRATION STORAGE ---
        self.calibrated_ear_threshold = 0.19
        self.calibrated_mar_baseline = 0.40
        self.calibrated_mar_yawn_threshold = 0.56
        self.calibrated_face_size = 100.0
        self.ear_std_dev = 0.05
        self.mar_std_dev = 0.12
        self.is_calibrating = False
        self.calibration_ear_values = []
        self.calibration_mar_values = []
        self.last_recalibration_time = None

        # --- DETECTION COUNTERS ---
        self.eye_closure_start_time = None
        self.yawn_start_time = None
        self.yawn_frame_counter = 0
        self.yawn_count = 0
        self.nod_count = 0
        self.blink_counter = 0
        self.blink_frame_counter = 0
        self.display_blink_rate = 0
        self.blink_analysis_start_time = time.time()
        self.is_yawning = False
        self.is_nodding = False
        self.fatigue_events = []

        # Rolling 60-second blink counter
        self.blink_timestamps = deque(maxlen=1000)  # Store last 1000 blink timestamps
        self.blinks_last_60s = 0

        # --- ROBUST YAWN STATE ---
        self.yawn_state = "IDLE"  # States: IDLE, OPENING, PEAK, CLOSING
        self.yawn_robust_start_time = None
        self.yawn_peak_mar = 0.0
        self.yawn_cooldown_counter = 0
        self.previous_mar = 0.0
        self.mar_rate_history = deque(maxlen=5)
        self.yawn_phase = "idle"

        # --- BUFFERS ---
        self.ear_history = deque(maxlen=self.SMOOTH_WINDOW)
        self.mar_history = deque(maxlen=self.SMOOTH_WINDOW)
        self.head_positions = deque(maxlen=self.HEAD_NOD_FRAME_WINDOW)
        self.last_face_size = None

        # --- FACE TRACKING FOR AUTO-RECALIBRATION ---
        self.face_present = False  # Track if face is currently detected
        self.face_lost_time = None  # Timestamp when face was lost
        self.face_detected_time = None  # Timestamp when face was first detected
        self.face_confirmed_active = False  # Track if face has been present for 2+ seconds

        # --- DLIB INIT ---
        if not os.path.exists(self.DLIB_LANDMARK_MODEL):
            raise FileNotFoundError(f"[ERROR] Download shape_predictor_68_face_landmarks.dat")

        print("[INFO] Loading facial landmark predictor...")
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.DLIB_LANDMARK_MODEL)
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        (self.mStart, self.mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
        (self.nStart, self.nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]

        # --- IMAGE PREPROCESSING FOR BACKLIGHT COMPENSATION ---
        # Reduced clipLimit to avoid over-enhancement and pixelation
        self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))

    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C) if C > 0 else 0.25

    def mouth_aspect_ratio(self, mouth):
        A = dist.euclidean(mouth[2], mouth[10])
        B = dist.euclidean(mouth[4], mouth[8])
        C = dist.euclidean(mouth[0], mouth[6])
        return (A + B) / (2.0 * C) if C > 0 else 0.35

    def remove_outliers(self, values):
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
        history.append(value)
        return np.mean(history) if len(history) > 0 else value

    def save_config(self):
        """Save current configuration to JSON file"""
        config = {
            "EAR_PERCENTILE": self.EAR_PERCENTILE,
            "EAR_PERCENTAGE_THRESHOLD": self.EAR_PERCENTAGE_THRESHOLD,
            "EYE_CLOSURE_SECONDS_THRESHOLD": self.EYE_CLOSURE_SECONDS_THRESHOLD,
            "YAWN_MAR_MULTIPLIER": self.YAWN_MAR_MULTIPLIER,
            "YAWN_DURATION_SECONDS": self.YAWN_DURATION_SECONDS,
            "YAWN_MIN_FRAMES": self.YAWN_MIN_FRAMES,
            "YAWN_MIN_DURATION": self.YAWN_MIN_DURATION,
            "YAWN_MAX_DURATION": self.YAWN_MAX_DURATION,
            "YAWN_OPENING_RATE_THRESH": self.YAWN_OPENING_RATE_THRESH,
            "YAWN_MAR_INCREASE_THRESH": self.YAWN_MAR_INCREASE_THRESH,
            "YAWN_COOLDOWN_FRAMES": self.YAWN_COOLDOWN_FRAMES,
            "BLINK_CONSEC_FRAMES": self.BLINK_CONSEC_FRAMES,
            "FATIGUE_BLINK_RATE_SECONDS": self.FATIGUE_BLINK_RATE_SECONDS,
            "FATIGUE_LOW_BLINK_THRESHOLD": self.FATIGUE_LOW_BLINK_THRESHOLD,
            "FATIGUE_HIGH_BLINK_THRESHOLD": self.FATIGUE_HIGH_BLINK_THRESHOLD,
            "DROWSY_MIN_YAWNS": self.DROWSY_MIN_YAWNS,
            "MIN_CONFIDENCE": self.MIN_CONFIDENCE,
            "FATIGUE_EVENT_THRESHOLD": self.FATIGUE_EVENT_THRESHOLD,
            "DROWSY_MIN_FATIGUE_EVENTS": self.DROWSY_MIN_FATIGUE_EVENTS,
            "AUTO_RECALIBRATE_SECONDS": self.AUTO_RECALIBRATE_SECONDS,
            "FACE_LOST_RECALIBRATE_SECONDS": self.FACE_LOST_RECALIBRATE_SECONDS,
            "FACE_ACTIVE_THRESHOLD_SECONDS": self.FACE_ACTIVE_THRESHOLD_SECONDS
        }
        try:
            with open(self.CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=4)
            print(f"[INFO] Saved config to {self.CONFIG_FILE}")
            return True
        except Exception as e:
            print(f"[ERROR] Could not save config: {e}")
            return False

    def preprocess_frame(self, gray, fast_mode=False):
        """
        Apply advanced preprocessing to handle backlighting and improve face detection.
        Uses CLAHE (Contrast Limited Adaptive Histogram Equalization) and gamma correction.
        fast_mode: If True, skip expensive operations for faster processing during calibration
        """
        # Always apply CLAHE for adaptive contrast enhancement
        clahe_applied = self.clahe.apply(gray)

        # Check if we need backlight compensation (adaptive filter selection)
        mean_brightness = np.mean(clahe_applied)
        needs_gamma = mean_brightness < 100  # Dark image needs brightening

        if needs_gamma:
            # Apply gamma correction for backlight compensation
            # Gamma < 1 brightens darker regions (helps with backlit faces)
            gamma = 0.7
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
            processed = cv2.LUT(clahe_applied, table)
        else:
            processed = clahe_applied

        # Skip bilateral filter in fast mode (calibration)
        if not fast_mode:
            # Apply bilateral filter to reduce noise while preserving edges
            # Reduced parameters to avoid over-smoothing and pixelation
            # d=3 (was 5), sigmaColor=30 (was 50), sigmaSpace=30 (was 50)
            processed = cv2.bilateralFilter(processed, 3, 30, 30)

        return processed

    def calculate_confidence(self, ear, mar, face_size):
        confidence = 0.0

        if self.calibrated_ear_threshold and self.ear_std_dev > 0:
            if ear < self.calibrated_ear_threshold:
                z_score = abs(self.calibrated_ear_threshold - ear) / self.ear_std_dev
                confidence += min(1.0, z_score / 3.0) * 0.4

        if self.calibrated_mar_baseline and self.mar_std_dev > 0:
            mar_deviation = abs(mar - self.calibrated_mar_baseline) / self.mar_std_dev
            if mar_deviation > 2.0:
                confidence += min(1.0, mar_deviation / 4.0) * 0.3

        if self.calibrated_face_size and face_size:
            size_ratio = face_size / self.calibrated_face_size
            if 0.6 < size_ratio < 1.4:
                confidence += 0.3

        return min(1.0, confidence)

    def detect_yawn_robust(self, current_mar, frame_time):
        """
        Robust yawn detection using state machine with temporal validation.
        Returns: (is_yawning, yawn_confidence, yawn_phase)
        """
        # Decrease cooldown counter
        if self.yawn_cooldown_counter > 0:
            self.yawn_cooldown_counter -= 1
            return False, 0.0, "cooldown"

        # Calculate MAR change rate
        mar_rate = current_mar - self.previous_mar
        self.mar_rate_history.append(mar_rate)
        self.previous_mar = current_mar

        # Define dynamic threshold based on calibrated baseline
        yawn_threshold = self.calibrated_mar_baseline * self.YAWN_MAR_INCREASE_THRESH

        is_yawning = False
        yawn_confidence = 0.0
        yawn_phase = self.yawn_state.lower()

        # State machine for yawn detection
        if self.yawn_state == "IDLE":
            # Check for mouth opening with sufficient rate
            if current_mar > yawn_threshold and mar_rate > self.YAWN_OPENING_RATE_THRESH:
                self.yawn_state = "OPENING"
                self.yawn_robust_start_time = frame_time
                self.yawn_peak_mar = current_mar

        elif self.yawn_state == "OPENING":
            # Track peak MAR
            if current_mar > self.yawn_peak_mar:
                self.yawn_peak_mar = current_mar

            # Check if mouth is still opening or at peak
            if current_mar > yawn_threshold:
                # Check if rate is slowing down (approaching peak)
                if abs(mar_rate) < self.YAWN_OPENING_RATE_THRESH * 0.5:
                    self.yawn_state = "PEAK"
            else:
                # Mouth closed too quickly - false positive
                self.yawn_state = "IDLE"
                self.yawn_robust_start_time = None

        elif self.yawn_state == "PEAK":
            # Update peak if still increasing
            if current_mar > self.yawn_peak_mar:
                self.yawn_peak_mar = current_mar

            # Check for closing motion (negative rate)
            if mar_rate < -self.YAWN_OPENING_RATE_THRESH * 0.3:
                self.yawn_state = "CLOSING"

            # Timeout if staying at peak too long
            if self.yawn_robust_start_time and (frame_time - self.yawn_robust_start_time) > self.YAWN_MAX_DURATION:
                self.yawn_state = "IDLE"
                self.yawn_robust_start_time = None

        elif self.yawn_state == "CLOSING":
            # Check if mouth has returned close to baseline
            if current_mar < self.calibrated_mar_baseline * 1.2:
                # Validate yawn duration
                if self.yawn_robust_start_time:
                    yawn_duration = frame_time - self.yawn_robust_start_time

                    if self.YAWN_MIN_DURATION <= yawn_duration <= self.YAWN_MAX_DURATION:
                        # Valid yawn detected!
                        is_yawning = True

                        # Calculate confidence based on duration and peak MAR
                        duration_score = min(1.0, yawn_duration / 3.0)
                        peak_score = min(1.0, (self.yawn_peak_mar / self.calibrated_mar_baseline - 1.0) / 0.5)
                        yawn_confidence = (duration_score * 0.6 + peak_score * 0.4)

                        # Set cooldown to prevent double-counting
                        self.yawn_cooldown_counter = self.YAWN_COOLDOWN_FRAMES

                # Reset state
                self.yawn_state = "IDLE"
                self.yawn_robust_start_time = None

            # Timeout check
            elif self.yawn_robust_start_time and (frame_time - self.yawn_robust_start_time) > self.YAWN_MAX_DURATION:
                self.yawn_state = "IDLE"
                self.yawn_robust_start_time = None

        self.yawn_phase = yawn_phase
        return is_yawning, yawn_confidence, yawn_phase

    def recalibrate(self, vs, reset_counters=False):
        """
        Recalibrate baseline values.

        Args:
            vs: Video stream
            reset_counters: If True, resets all counters and state to zero (for new person).
                          If False, only updates baselines (for periodic recalibration).
        """
        print(f"\n[{time.ctime()}] Starting calibration...")
        print("[INFO] Stay still, eyes open, mouth closed")
        if reset_counters:
            print("[INFO] This is a full recalibration - counters will be reset")
        else:
            print("[INFO] This is a periodic recalibration - counters will be preserved")

        self.is_calibrating = True
        self.calibration_ear_values = []
        self.calibration_mar_values = []
        face_sizes = []
        stable_frames = 0
        total_frames = 0

        while stable_frames < self.CALIBRATION_FRAMES and total_frames < self.CALIBRATION_FRAMES * 3:
            ret, frame = vs.read()
            if not ret:
                continue

            total_frames += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply preprocessing in fast mode for calibration (skips bilateral filter)
            gray_processed = self.preprocess_frame(gray, fast_mode=True)

            # Convert grayscale back to BGR for display with colored overlays
            frame = cv2.cvtColor(gray_processed, cv2.COLOR_GRAY2BGR)

            rects = self.detector(gray_processed, 0)  # Use 0 instead of 1 for faster detection

            progress = int((stable_frames / self.CALIBRATION_FRAMES) * 100)
            cv2.putText(frame, f"CALIBRATING: {progress}%", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"Samples: {stable_frames}/{self.CALIBRATION_FRAMES}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            if len(rects) > 0:
                rect = rects[0]
                face_width = rect.width()

                if face_width >= self.MIN_FACE_SIZE:
                    try:
                        shape = self.predictor(gray_processed, rect)
                        shape = face_utils.shape_to_np(shape)

                        leftEye = shape[self.lStart:self.lEnd]
                        rightEye = shape[self.rStart:self.rEnd]
                        mouth = shape[self.mStart:self.mEnd]

                        ear = (self.eye_aspect_ratio(leftEye) + self.eye_aspect_ratio(rightEye)) / 2.0
                        mar = self.mouth_aspect_ratio(mouth)

                        if 0.15 < ear < 0.45 and 0.20 < mar < 0.70:
                            if len(self.ear_history) > 0:
                                recent_avg = np.mean(list(self.ear_history))
                                if abs(ear - recent_avg) / max(recent_avg, 0.01) < 0.30:
                                    self.calibration_ear_values.append(ear)
                                    self.calibration_mar_values.append(mar)
                                    face_sizes.append(face_width)
                                    stable_frames += 1
                            else:
                                self.ear_history.append(ear)

                            cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 2)
                            cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 2)
                    except:
                        pass

            try:
                while not frame_queue.empty():
                    frame_queue.get_nowait()
                frame_queue.put(frame)
            except:
                pass

        if len(self.calibration_ear_values) >= self.CALIBRATION_FRAMES * 0.6:
            filtered_ears = self.remove_outliers(self.calibration_ear_values)
            filtered_mars = self.remove_outliers(self.calibration_mar_values)

            ear_percentile = np.percentile(filtered_ears, self.EAR_PERCENTILE)
            self.calibrated_ear_threshold = ear_percentile * self.EAR_PERCENTAGE_THRESHOLD
            self.ear_std_dev = np.std(filtered_ears) * 1.2

            self.calibrated_mar_baseline = np.median(filtered_mars)
            self.calibrated_mar_yawn_threshold = self.calibrated_mar_baseline * self.YAWN_MAR_MULTIPLIER
            self.mar_std_dev = np.std(filtered_mars) * 1.2

            self.calibrated_face_size = np.median(face_sizes)

            print(f"\nCalibration Success!")
            print(f"  EAR: {self.calibrated_ear_threshold:.3f}")
            print(f"  MAR: {self.calibrated_mar_baseline:.3f}")
            print(f"  Yawn Threshold: {self.calibrated_mar_yawn_threshold:.3f}\n")
        else:
            print(f"\n[WARNING] Using fallback values")

        self.is_calibrating = False
        self.last_recalibration_time = time.time()

        # Only reset counters if requested (new person detected)
        if reset_counters:
            self.reset_counters()

        # Always clear histories to adapt to new baselines
        self.ear_history.clear()
        self.mar_history.clear()

    def reset_counters(self):
        """Reset all counters, state, and detection values after recalibration"""
        self.eye_closure_start_time = None
        self.yawn_start_time = None
        self.yawn_frame_counter = 0
        self.yawn_count = 0
        self.nod_count = 0
        self.blink_counter = 0
        self.blink_frame_counter = 0
        self.display_blink_rate = 0
        self.blink_analysis_start_time = time.time()
        self.blink_timestamps.clear()
        self.blinks_last_60s = 0
        self.fatigue_events = []
        self.state_counter = 0
        self.confidence_score = 0.0
        self.current_state = self.STATE["ACTIVE"]
        self.previous_state = self.STATE["ACTIVE"]

        # Reset yawn detection state
        self.yawn_state = "IDLE"
        self.yawn_robust_start_time = None
        self.yawn_peak_mar = 0.0
        self.yawn_cooldown_counter = 0
        self.yawn_phase = "idle"

        # Reset head position tracking
        self.head_positions.clear()

    def run(self):
        print("\n" + "=" * 70)
        print("  SIMPLE BUS DROWSINESS DETECTION")
        print("=" * 70)
        print("  Camera: Side-mounted")
        print("  Features: Blinks + Yawns + Nods + Fatigue")
        print("  Web UI: http://localhost:5000")
        print("=" * 70 + "\n")

        # Configure VideoCapture for optimal RTSP streaming
        if self.USE_RTSP:
            print(f"[INFO] Connecting to RTSP stream: {self.RTSP_URL}")

            # Build RTSP URL with UDP transport and low latency options
            # UDP is faster than TCP and has lower latency
            rtsp_url_with_options = self.CAMERA_SOURCE

            # Use GStreamer pipeline for better RTSP handling (more reliable than FFMPEG for IP cameras)
            # This pipeline handles network issues, packet loss, and provides smooth playback
            gst_pipeline = (
                f"rtspsrc location={self.CAMERA_SOURCE} latency=0 protocols=udp ! "
                "rtph264depay ! h264parse ! avdec_h264 ! "
                "videoconvert ! video/x-raw,format=BGR ! appsink drop=1 sync=0"
            )

            print("[INFO] Using GStreamer pipeline for optimal RTSP performance")
            print("[INFO] - Protocol: UDP (lower latency than TCP)")
            print("[INFO] - Latency: 0ms buffer")
            print("[INFO] - Frame dropping enabled (prevents buffer buildup)")

            vs = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

            # Fallback to FFMPEG if GStreamer fails
            if not vs.isOpened():
                print("[WARNING] GStreamer failed, falling back to FFMPEG with UDP...")
                # Add RTSP options for FFMPEG to use UDP and reduce latency
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                    "rtsp_transport;udp|"
                    "fflags;nobuffer|"
                    "flags;low_delay|"
                    "framedrop;1|"
                    "max_delay;0"
                )
                vs = cv2.VideoCapture(self.CAMERA_SOURCE, cv2.CAP_FFMPEG)
                vs.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            print(f"[INFO] Using webcam: {self.CAMERA_SOURCE}")
            vs = cv2.VideoCapture(self.CAMERA_SOURCE)

            # Set maximum resolution and framerate for webcam
            vs.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            vs.set(cv2.CAP_PROP_FPS, 60)

        # Check if camera opened successfully
        if not vs.isOpened():
            raise RuntimeError(f"[ERROR] Failed to open camera: {self.CAMERA_SOURCE}")

        # Get actual values that camera supports
        actual_width = vs.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = vs.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = vs.get(cv2.CAP_PROP_FPS)
        print(f"[INFO] Camera resolution: {int(actual_width)}x{int(actual_height)} @ {actual_fps} FPS")

        if self.USE_RTSP:
            print("[INFO] RTSP stream configured for low-latency operation")

        time.sleep(1.0)  # Reduced from 2.0 for faster startup

        # Initial calibration - always reset counters
        self.recalibrate(vs, reset_counters=True)

        fps_counter = 0
        fps_start_time = time.time()
        fps = 0

        while True:
            # Manual recalibration - reset counters
            global recalibrate_flag
            if recalibrate_flag:
                print("\n[INFO] Manual recalibration (from UI button)...")
                self.recalibrate(vs, reset_counters=True)
                recalibrate_flag = False

            # Auto recalibration (periodic) - do NOT reset counters, only update baselines
            if (not self.is_calibrating and self.last_recalibration_time and
                    time.time() - self.last_recalibration_time > self.AUTO_RECALIBRATE_SECONDS):
                print(f"\n[INFO] Periodic auto-recalibration (every {self.AUTO_RECALIBRATE_SECONDS}s)...")
                self.recalibrate(vs, reset_counters=False)

            ret, frame = vs.read()
            if not ret or frame is None:
                # For RTSP streams, reconnect on failure
                if self.USE_RTSP:
                    print("[WARNING] RTSP stream read failed. Attempting reconnect...")
                    vs.release()
                    time.sleep(0.5)  # Short delay before reconnect

                    # Try GStreamer pipeline first
                    gst_pipeline = (
                        f"rtspsrc location={self.CAMERA_SOURCE} latency=0 protocols=udp ! "
                        "rtph264depay ! h264parse ! avdec_h264 ! "
                        "videoconvert ! video/x-raw,format=BGR ! appsink drop=1 sync=0"
                    )
                    vs = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

                    # Fallback to FFMPEG if GStreamer fails
                    if not vs.isOpened():
                        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                            "rtsp_transport;udp|fflags;nobuffer|flags;low_delay|framedrop;1|max_delay;0"
                        )
                        vs = cv2.VideoCapture(self.CAMERA_SOURCE, cv2.CAP_FFMPEG)
                        vs.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                    print("[INFO] RTSP stream reconnected")
                    continue
                else:
                    break

            # Validate frame integrity (skip corrupted frames from network issues)
            if frame.size == 0 or frame.shape[0] < 100 or frame.shape[1] < 100:
                print("[WARNING] Corrupted/invalid frame detected, skipping...")
                continue

            # Additional validation: check if frame is all black or all white (corruption indicator)
            frame_mean = np.mean(frame)
            if frame_mean < 5 or frame_mean > 250:
                print("[WARNING] Abnormal frame detected (all black/white), skipping...")
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply preprocessing for better face detection with backlighting
            gray_processed = self.preprocess_frame(gray)

            # Convert grayscale back to BGR for display with colored overlays
            frame = cv2.cvtColor(gray_processed, cv2.COLOR_GRAY2BGR)

            fps_counter += 1
            if fps_counter % 15 == 0:
                fps = 15 / (time.time() - fps_start_time)
                fps_start_time = time.time()

            # Run face detection every 3rd frame for better performance (from every 2nd)
            # This improves FPS from ~20 to ~27-30 FPS
            if fps_counter % 3 == 0:
                rects = self.detector(gray_processed, 0)
            else:
                rects = getattr(self, 'last_rects', [])

            status_text = "No Face"
            color = (128, 128, 128)
            current_time = time.time()

            # Track if we have a valid face with landmarks this frame
            valid_face_detected = False

            # Check for multiple faces - reject if more than one detected
            if len(rects) > 1:
                status_text = "Multiple Faces Detected"
                color = (255, 165, 0)  # Orange
                # Draw rectangles around all detected faces
                for rect in rects:
                    (x, y, w, h) = (rect.left(), rect.top(), rect.width(), rect.height())
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
                    cv2.putText(frame, "MULTIPLE FACES", (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                # Don't process - wait for single person
                valid_face_detected = False
            elif len(rects) == 1:
                self.last_rects = rects  # Cache for next frame
                largest_rect = rects[0]  # Only one face
                face_width = largest_rect.width()

                if face_width >= self.MIN_FACE_SIZE and not self.is_calibrating:
                    (x, y, w, h) = (largest_rect.left(), largest_rect.top(),
                                    largest_rect.width(), largest_rect.height())
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    try:
                        shape = self.predictor(gray_processed, largest_rect)
                        shape = face_utils.shape_to_np(shape)

                        leftEye = shape[self.lStart:self.lEnd]
                        rightEye = shape[self.rStart:self.rEnd]
                        mouth = shape[self.mStart:self.mEnd]
                        nose = shape[self.nStart:self.nEnd]

                        ear_raw = (self.eye_aspect_ratio(leftEye) + self.eye_aspect_ratio(rightEye)) / 2.0
                        mar_raw = self.mouth_aspect_ratio(mouth)

                        ear = self.apply_smoothing(ear_raw, self.ear_history)
                        mar = self.apply_smoothing(mar_raw, self.mar_history)

                        self.confidence_score = self.calculate_confidence(ear, mar, face_width)

                        is_eyes_closed = ear < self.calibrated_ear_threshold
                        is_mouth_open_wide = mar > self.calibrated_mar_yawn_threshold

                        # BLINK DETECTION
                        current_time = time.time()
                        if is_eyes_closed:
                            self.blink_frame_counter += 1
                        else:
                            if self.blink_frame_counter >= self.BLINK_CONSEC_FRAMES:
                                self.blink_counter += 1
                                self.blink_timestamps.append(current_time)  # Record blink timestamp
                            self.blink_frame_counter = 0

                        # Calculate blinks in last 60 seconds (rolling counter)
                        cutoff_time = current_time - 60.0
                        self.blinks_last_60s = sum(1 for ts in self.blink_timestamps if ts >= cutoff_time)

                        fatigue_detected = False
                        if current_time - self.blink_analysis_start_time > self.FATIGUE_BLINK_RATE_SECONDS:
                            self.display_blink_rate = (self.blink_counter / self.FATIGUE_BLINK_RATE_SECONDS) * 60
                            if (self.display_blink_rate < self.FATIGUE_LOW_BLINK_THRESHOLD or
                                    self.display_blink_rate > self.FATIGUE_HIGH_BLINK_THRESHOLD):
                                if self.confidence_score > 0.3:
                                    fatigue_detected = True
                                    self.fatigue_events.append(current_time)
                            self.blink_counter = 0
                            self.blink_analysis_start_time = current_time

                        self.fatigue_events = [e for e in self.fatigue_events
                                               if time.time() - e < self.FATIGUE_EVENT_WINDOW_SECONDS]
                        if len(self.fatigue_events) >= self.FATIGUE_EVENT_THRESHOLD:
                            fatigue_detected = True

                        # ROBUST YAWN DETECTION
                        drowsiness_event = False
                        current_time = time.time()
                        yawn_detected_robust, yawn_conf, yawn_phase = self.detect_yawn_robust(mar, current_time)

                        if yawn_detected_robust:
                            self.yawn_count += 1
                            self.fatigue_events.append(current_time)
                            if self.yawn_count >= self.DROWSY_MIN_YAWNS and self.current_state == self.STATE["FATIGUED"]:
                                drowsiness_event = True

                        # HEAD NOD DETECTION
                        if not is_mouth_open_wide:
                            nose_tip_y = np.mean([p[1] for p in nose])
                            self.head_positions.append(nose_tip_y)
                            if len(self.head_positions) >= self.HEAD_NOD_FRAME_WINDOW:
                                positions = np.array(self.head_positions)
                                kernel = np.ones(self.NOD_SMOOTHING_KERNEL) / self.NOD_SMOOTHING_KERNEL
                                smoothed = np.convolve(positions, kernel, mode='valid')
                                if len(smoothed) > 20:
                                    nod_threshold = self.calibrated_face_size * self.HEAD_NOD_THRESHOLD_MULTIPLIER
                                    movement = max(smoothed) - smoothed[-1]
                                    if movement > nod_threshold * 1.5:
                                        if not self.is_nodding:
                                            self.nod_count += 1
                                            self.fatigue_events.append(time.time())
                                            self.is_nodding = True
                                        if self.nod_count >= 8 and self.current_state == self.STATE["FATIGUED"]:
                                            drowsiness_event = True
                                    else:
                                        self.is_nodding = False

                        # EYE CLOSURE
                        if is_eyes_closed and self.confidence_score > self.MIN_CONFIDENCE * 0.8:
                            if self.eye_closure_start_time is None:
                                self.eye_closure_start_time = time.time()
                            else:
                                closure_duration = time.time() - self.eye_closure_start_time
                                if closure_duration >= self.DROWSY_MIN_EYE_CLOSURE_DURATION and self.current_state == \
                                        self.STATE["FATIGUED"]:
                                    drowsiness_event = True
                        else:
                            self.eye_closure_start_time = None

                        # STATE MACHINE
                        target_state = self.current_state

                        if self.current_state == self.STATE["ACTIVE"]:
                            if fatigue_detected and len(self.fatigue_events) >= self.FATIGUE_EVENT_THRESHOLD:
                                target_state = self.STATE["FATIGUED"]

                        elif self.current_state == self.STATE["FATIGUED"]:
                            has_many_events = len(self.fatigue_events) >= self.DROWSY_MIN_FATIGUE_EVENTS
                            has_enough_yawns = self.yawn_count >= self.DROWSY_MIN_YAWNS

                            if has_many_events and (
                                    drowsiness_event or has_enough_yawns) and self.confidence_score > self.MIN_CONFIDENCE * 0.8:
                                target_state = self.STATE["DROWSY"]
                            elif not fatigue_detected or len(self.fatigue_events) < 3:
                                target_state = self.STATE["ACTIVE"]

                        elif self.current_state == self.STATE["DROWSY"]:
                            if not drowsiness_event and len(self.fatigue_events) < 6:
                                target_state = self.STATE["FATIGUED"]

                        # Transitions
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

                        # Status - only show "Active" if face has been confirmed active for 2+ seconds
                        if not self.face_confirmed_active:
                            status_text = "Detecting Face..."
                            color = (255, 255, 0)  # Yellow
                        elif self.current_state == self.STATE["DROWSY"]:
                            status_text = "DROWSINESS ALERT!"
                            color = (0, 0, 255)
                        elif self.current_state == self.STATE["FATIGUED"]:
                            status_text = "Fatigue Detected"
                            color = (0, 165, 255)
                        else:
                            status_text = "Active"
                            color = (0, 255, 0)

                        # Draw eyes with color change on blink
                        if is_eyes_closed:
                            eye_color = (0, 0, 255)  # Red when eyes closed
                            eye_thickness = 2
                        else:
                            eye_color = (0, 255, 0)  # Green when eyes open
                            eye_thickness = 1
                        cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, eye_color, eye_thickness)
                        cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, eye_color, eye_thickness)

                        # Color mouth based on yawn phase
                        if yawn_phase == "opening":
                            mouth_color = (0, 255, 255)  # Yellow
                            mouth_thickness = 2
                        elif yawn_phase == "peak":
                            mouth_color = (0, 165, 255)  # Orange
                            mouth_thickness = 2
                        elif yawn_phase == "closing":
                            mouth_color = (255, 0, 255)  # Magenta
                            mouth_thickness = 2
                        elif yawn_detected_robust:
                            mouth_color = (0, 0, 255)  # Red
                            mouth_thickness = 2
                        else:
                            mouth_color = (0, 255, 0)  # Green
                            mouth_thickness = 1
                        cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, mouth_color, mouth_thickness)

                        # Mark that we successfully detected valid face with landmarks
                        valid_face_detected = True

                    except:
                        status_text = "Processing..."
                        color = (255, 255, 0)
                        valid_face_detected = False

            # Face tracking logic - only track if we have valid face with landmarks
            if valid_face_detected:
                # Valid face with landmarks detected
                if not self.face_present:
                    # Face just appeared - start detection timer
                    self.face_detected_time = current_time
                    self.face_present = True
                    self.face_confirmed_active = False
                    print(f"\n[INFO] Valid face detected. Waiting for {self.FACE_ACTIVE_THRESHOLD_SECONDS}s to confirm active state...")

                # Check if face has been present long enough to be "active"
                if not self.face_confirmed_active and self.face_detected_time is not None:
                    face_present_duration = current_time - self.face_detected_time
                    if face_present_duration >= self.FACE_ACTIVE_THRESHOLD_SECONDS:
                        # Face has been present for threshold seconds - now it's "active"
                        self.face_confirmed_active = True
                        print(f"\n[INFO] Face confirmed active after {face_present_duration:.1f}s.")

                        # Check if we need to recalibrate (face was previously lost for threshold duration)
                        if self.face_lost_time is not None and not self.is_calibrating:
                            face_lost_duration = self.face_detected_time - self.face_lost_time
                            if face_lost_duration >= self.FACE_LOST_RECALIBRATE_SECONDS:
                                print(f"\n[INFO] Previous face was lost for {face_lost_duration:.1f}s. Auto-recalibrating for new person...")
                                self.recalibrate(vs, reset_counters=True)
                            else:
                                print(f"\n[INFO] Previous face was lost for {face_lost_duration:.1f}s (< {self.FACE_LOST_RECALIBRATE_SECONDS}s threshold). Continuing without recalibration.")

                        self.face_lost_time = None
            else:
                # No valid face detected
                if self.face_present:
                    # Face just disappeared - record the time
                    if self.face_confirmed_active:
                        print("\n[INFO] Active face lost from frame. Recording time...")
                        self.face_lost_time = current_time
                    else:
                        print("\n[INFO] Face lost before becoming active.")

                    self.face_present = False
                    self.face_detected_time = None
                    self.face_confirmed_active = False

            # Only display status text on video feed (no other text)
            # All values will be shown on the webpage instead

            # Calculate additional values for display
            time_to_check = max(0, self.FATIGUE_BLINK_RATE_SECONDS - (time.time() - self.blink_analysis_start_time))

            # Get MAR and thresholds
            current_mar = mar if 'mar' in locals() else 0.0
            current_ear = ear if 'ear' in locals() else 0.0
            yawn_thresh = self.calibrated_mar_baseline * self.YAWN_MAR_INCREASE_THRESH if self.calibrated_mar_baseline > 0 else 0.0
            ear_thresh = self.calibrated_ear_threshold if self.calibrated_ear_threshold > 0 else 0.0

            # Calculate state reasons (ensure all values are JSON serializable)
            state_reasons = {
                "fatigue_events": int(len(self.fatigue_events)),
                "fatigue_threshold": int(self.FATIGUE_EVENT_THRESHOLD),
                "drowsy_fatigue_threshold": int(self.DROWSY_MIN_FATIGUE_EVENTS),
                "yawn_count": int(self.yawn_count),
                "yawn_threshold": int(self.DROWSY_MIN_YAWNS),
                "is_eyes_closed": bool(is_eyes_closed) if 'is_eyes_closed' in locals() else False,
                "eye_closure_duration": float((time.time() - self.eye_closure_start_time) if self.eye_closure_start_time else 0.0),
                "eye_closure_threshold": float(self.DROWSY_MIN_EYE_CLOSURE_DURATION),
                "blink_rate": float(self.display_blink_rate),
                "blink_low_threshold": int(self.FATIGUE_LOW_BLINK_THRESHOLD),
                "blink_high_threshold": int(self.FATIGUE_HIGH_BLINK_THRESHOLD),
                "current_state": int(self.current_state),
                "target_state": int(target_state if 'target_state' in locals() else self.current_state),
                "fatigue_detected": bool(fatigue_detected if 'fatigue_detected' in locals() else False),
                "drowsiness_event": bool(drowsiness_event if 'drowsiness_event' in locals() else False)
            }

            # Update status
            global current_status
            current_status = {
                "status": status_text,
                "confidence": float(self.confidence_score),
                "blink_rate": float(self.display_blink_rate),
                "blinks_last_60s": int(self.blinks_last_60s),
                "yawns": int(self.yawn_count),
                "nods": int(self.nod_count),
                "fatigue_check": int(time_to_check),
                "yawn_phase": self.yawn_phase if hasattr(self, 'yawn_phase') else "idle",
                "fps": float(fps),
                "mar": float(current_mar),
                "mar_threshold": float(yawn_thresh),
                "ear": float(current_ear),
                "ear_threshold": float(ear_thresh),
                "state_reasons": state_reasons
            }

            try:
                while not frame_queue.empty():
                    frame_queue.get_nowait()
                frame_queue.put(frame)
            except:
                pass

        vs.release()


@app.route('/')
def index():
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>Bus Driver Drowsiness Detection</title>
    <meta charset="UTF-8">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: Arial, sans-serif;
            background: #1a1a1a;
            color: #ffffff;
            padding: 8px;
            overflow: hidden;
            height: 100vh;
        }
        .container {
            max-width: 100%;
            height: 100%;
            display: flex;
            flex-direction: column;
        }
        .main-grid {
            display: grid;
            grid-template-columns: 1.8fr 1fr;
            gap: 10px;
            flex: 1;
            overflow: hidden;
        }
        h1 {
            text-align: center;
            font-size: 1.3em;
            color: #7fff00;
            margin-bottom: 8px;
            font-weight: normal;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            gap: 6px;
            margin-bottom: 8px;
        }
        .stat-box {
            background: #2a2a2a;
            padding: 6px;
            border-radius: 6px;
            text-align: center;
        }
        .stat-box h3 {
            font-size: 0.65em;
            color: #ffffff;
            font-weight: normal;
            margin-bottom: 2px;
        }
        .stat-box h2 {
            font-size: 1em;
            margin: 0;
            font-weight: normal;
        }
        .status-active { color: #7fff00; }
        .status-warning { color: #ffaa00; }
        .status-alert { color: #ff3333; animation: pulse 1s infinite; }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.6; }
        }
        .left-panel {
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .video-container {
            border-radius: 6px;
            overflow: hidden;
            border: 2px solid #333;
            background: #000;
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .video-container img { max-width: 100%; max-height: 100%; object-fit: contain; display: block; }
        .controls {
            text-align: center;
            margin: 8px 0;
        }
        button {
            background: #7fff00;
            color: #000;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-weight: bold;
            font-size: 0.85em;
            transition: all 0.3s;
        }
        button:hover {
            background: #6fd900;
            transform: translateY(-1px);
        }
        .right-panel {
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .state-section {
            background: #2a2a2a;
            padding: 10px;
            border-radius: 6px;
            margin-bottom: 8px;
            flex-shrink: 0;
        }
        .config-section {
            background: #2a2a2a;
            padding: 10px;
            border-radius: 6px;
            flex: 1;
            overflow-y: auto;
            overflow-x: hidden;
            display: flex;
            flex-direction: column;
        }
        .config-section::-webkit-scrollbar {
            width: 6px;
        }
        .config-section::-webkit-scrollbar-track {
            background: #1a1a1a;
        }
        .config-section::-webkit-scrollbar-thumb {
            background: #7fff00;
            border-radius: 3px;
        }
        .config-section h2, .state-section h2 {
            font-size: 0.95em;
            margin-bottom: 8px;
            color: #7fff00;
        }
        .config-input {
            display: grid;
            grid-template-columns: 1.5fr 1fr;
            gap: 6px;
            margin-bottom: 6px;
            align-items: center;
        }
        .config-input label {
            font-size: 0.7em;
        }
        .input-with-arrows {
            display: flex;
            gap: 4px;
            align-items: center;
        }
        .config-input input {
            padding: 4px;
            background: #1a1a1a;
            border: 1px solid #444;
            color: #fff;
            border-radius: 3px;
            flex: 1;
            text-align: center;
            font-size: 0.75em;
        }
        .arrow-btn {
            background: #3a3a3a;
            border: 1px solid #555;
            color: #7fff00;
            padding: 2px 6px;
            border-radius: 3px;
            cursor: pointer;
            font-size: 0.75em;
            transition: background 0.2s;
        }
        .arrow-btn:hover {
            background: #4a4a4a;
        }
        .arrow-btn:active {
            background: #2a2a2a;
        }
        .state-formula {
            background: #1a1a1a;
            padding: 5px;
            border-radius: 3px;
            margin: 3px 0;
            font-size: 0.68em;
            line-height: 1.3;
        }
        .counter-display {
            background: #1a1a1a;
            padding: 8px;
            border-radius: 4px;
            text-align: center;
            margin-top: 6px;
        }
        .counter-display h3 {
            font-size: 0.75em;
            margin-bottom: 4px;
            color: #888;
        }
        .counter-display .counter-value {
            font-size: 1.6em;
            color: #7fff00;
            font-weight: bold;
        }
        .state-value {
            display: inline-block;
            padding: 1px 4px;
            border-radius: 2px;
            margin: 0 1px;
            font-size: 0.95em;
        }
        .value-normal { background: #2a4a2a; color: #7fff00; }
        .value-trigger { background: #4a2a2a; color: #ff6666; font-weight: bold; }
        button.save-btn {
            width: 100%;
            margin-top: 6px;
            font-size: 0.8em;
            padding: 6px 12px;
        }
        .config-section h2 {
            font-size: 1.1em;
        }
        .description-box {
            background: #1a3a1a;
            padding: 8px;
            border-radius: 4px;
            margin-top: 8px;
            border: 1px solid #3a5a3a;
        }
        .description-box strong {
            font-size: 0.75em;
            color: #7fff00;
            display: block;
            margin-bottom: 4px;
        }
        .description-box p {
            font-size: 0.65em;
            line-height: 1.3;
            color: #aaa;
            margin: 0;
        }
    </style>
    <script>
        let currentConfig = {};

        function updateStatus() {
            fetch('/status')
                .then(r => r.json())
                .then(d => {
                    const s = document.getElementById('status');
                    s.textContent = d.status;
                    s.className = d.status.includes('ALERT') ? 'status-alert' :
                                  d.status.includes('Fatigue') ? 'status-warning' : 'status-active';
                    document.getElementById('confidence').textContent = d.confidence.toFixed(2);
                    document.getElementById('blinks_last_60s').textContent = d.blinks_last_60s || 0;
                    document.getElementById('yawns').textContent = d.yawns;
                    document.getElementById('nods').textContent = d.nods;
                    document.getElementById('fatigue_check').textContent = d.fatigue_check + 's';
                    document.getElementById('yawn_phase').textContent = d.yawn_phase || 'idle';
                    document.getElementById('fps').textContent = d.fps ? d.fps.toFixed(1) : '0';
                    document.getElementById('ear').textContent = d.ear ? d.ear.toFixed(3) : '0.000';
                    document.getElementById('ear_threshold').textContent = d.ear_threshold ? d.ear_threshold.toFixed(3) : '0.000';
                    document.getElementById('mar').textContent = d.mar ? d.mar.toFixed(3) : '0.000';
                    document.getElementById('mar_threshold').textContent = d.mar_threshold ? d.mar_threshold.toFixed(3) : '0.000';

                    // Update state formulas
                    if(d.state_reasons) updateStateFormulas(d.state_reasons);
                });
        }

        function updateStateFormulas(reasons) {
            const state = reasons.current_state;
            const fe = reasons.fatigue_events;
            const ft = reasons.fatigue_threshold;
            const dft = reasons.drowsy_fatigue_threshold;
            const yc = reasons.yawn_count;
            const yt = reasons.yawn_threshold;
            const br = reasons.blink_rate;
            const blt = reasons.blink_low_threshold;
            const bht = reasons.blink_high_threshold;

            // Update fatigue events counter
            document.getElementById('fatigue_events_count').textContent = fe;

            // ACTIVE -> FATIGUED
            document.getElementById('formula_active').innerHTML =
                `Fatigue Events <span class="${fe >= ft ? 'value-trigger' : 'value-normal'}">${fe}</span> >= <span class="value-normal">${ft}</span>`;

            // FATIGUED -> DROWSY
            document.getElementById('formula_fatigued').innerHTML =
                `Fatigue Events <span class="${fe >= dft ? 'value-trigger' : 'value-normal'}">${fe}</span> >= <span class="value-normal">${dft}</span> AND
                Yawns <span class="${yc >= yt ? 'value-trigger' : 'value-normal'}">${yc}</span> >= <span class="value-normal">${yt}</span>`;

            // Fatigue Triggers
            const blinkTrigger = br < blt || br > bht;
            document.getElementById('fatigue_triggers').innerHTML =
                `Blink Rate <span class="${blinkTrigger ? 'value-trigger' : 'value-normal'}">${br.toFixed(0)}</span>
                (< <span class="value-normal">${blt}</span> or > <span class="value-normal">${bht}</span>)`;
        }

        function loadConfig() {
            fetch('/get_config')
                .then(r => r.json())
                .then(config => {
                    currentConfig = config;
                    Object.keys(config).forEach(key => {
                        const input = document.getElementById(key);
                        if(input) input.value = config[key];
                    });
                });
        }

        function saveConfig() {
            const config = {};
            Object.keys(currentConfig).forEach(key => {
                const input = document.getElementById(key);
                if(input) config[key] = parseFloat(input.value);
            });

            fetch('/save_config', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(config)
            })
            .then(r => r.json())
            .then(data => {
                alert(data.message);
                loadConfig();
            });
        }

        function adjustValue(inputId, delta) {
            const input = document.getElementById(inputId);
            if(!input) return;

            const step = parseFloat(input.step) || 0.01;
            const currentValue = parseFloat(input.value) || 0;
            const newValue = currentValue + (delta * step);

            // Respect min/max bounds
            const min = input.min ? parseFloat(input.min) : -Infinity;
            const max = input.max ? parseFloat(input.max) : Infinity;

            input.value = Math.max(min, Math.min(max, newValue)).toFixed(
                step < 1 ? (step.toString().split('.')[1] || '').length : 0
            );
        }

        function recalibrate() {
            fetch('/recalibrate', {method: 'POST'})
                .then(r => r.json())
                .then(data => {
                    document.getElementById('status').textContent = 'Calibrating...';
                    document.getElementById('status').className = 'status-warning';
                });
        }

        // Configuration descriptions
        const configDescriptions = {
            'EAR_PERCENTAGE_THRESHOLD': 'Eye Aspect Ratio threshold as percentage of calibrated baseline. Lower values = more sensitive to eye closure. Range: 0.0-1.0.',
            'EYE_CLOSURE_SECONDS_THRESHOLD': 'Duration (in seconds) eyes must be closed continuously to trigger drowsiness alert. Lower values = faster alerts.',
            'YAWN_MAR_INCREASE_THRESH': 'Mouth Aspect Ratio multiplier for yawn detection. How much mouth must open relative to baseline. Lower = more sensitive.',
            'YAWN_MIN_DURATION': 'Minimum duration (seconds) for valid yawn. Too low may trigger false positives. Typical yawn: 0.8-3 seconds.',
            'YAWN_OPENING_RATE_THRESH': 'Rate of mouth opening required to detect yawn. Lower = more sensitive to mouth movements.',
            'BLINK_CONSEC_FRAMES': 'Number of consecutive frames with closed eyes to count as blink. Higher = filters fast eye movements.',
            'FATIGUE_LOW_BLINK_THRESHOLD': 'Minimum blinks (per 60s) to avoid fatigue. Below this triggers low blink fatigue event.',
            'FATIGUE_HIGH_BLINK_THRESHOLD': 'Maximum blinks (per 60s) before excessive. Above this triggers high blink fatigue event.',
            'DROWSY_MIN_YAWNS': 'Number of yawns in recent history to trigger drowsy state. Lower = more sensitive to yawning.',
            'MIN_CONFIDENCE': 'Minimum confidence score for yawn detection. Higher = fewer false positives, but may miss genuine yawns.',
            'FATIGUE_EVENT_THRESHOLD': 'Number of recent fatigue events needed to transition to ACTIVE alert state. Lower = faster alerts.',
            'DROWSY_MIN_FATIGUE_EVENTS': 'Number of fatigue events required to escalate from ACTIVE to DROWSY state. Lower = escalates faster.',
            'AUTO_RECALIBRATE_SECONDS': 'Interval (seconds) between automatic recalibrations. Helps adapt to lighting changes. 30-300 recommended.',
            'FACE_LOST_RECALIBRATE_SECONDS': 'Duration (seconds) face must be lost before triggering recalibration when it returns. Useful for detecting person changes. 0.5-10 recommended.',
            'FATIGUE_BLINK_RATE_SECONDS': 'Time window (seconds) for blink rate analysis and "Check In" countdown. Blink rate is calculated over this period. 30-120 recommended.',
            'FACE_ACTIVE_THRESHOLD_SECONDS': 'Time (seconds) face must be continuously present before being considered "active". Prevents false detections from people walking by. 1-5 recommended.'
        };

        function showDescription(inputId) {
            const descText = document.getElementById('descriptionText');
            const desc = configDescriptions[inputId];
            if (desc) {
                descText.textContent = desc;
            }
        }

        function resetDescription() {
            const descText = document.getElementById('descriptionText');
            descText.textContent = 'Hover over or click on a configuration field to see its description.';
        }

        // Add event listeners to all config inputs
        function attachDescriptionListeners() {
            Object.keys(configDescriptions).forEach(inputId => {
                const input = document.getElementById(inputId);
                if (input) {
                    input.addEventListener('focus', () => showDescription(inputId));
                    input.addEventListener('mouseenter', () => showDescription(inputId));
                    input.addEventListener('blur', resetDescription);
                    input.addEventListener('mouseleave', resetDescription);
                }
            });
        }

        setInterval(updateStatus, 500);
        window.onload = () => {
            updateStatus();
            loadConfig();
            attachDescriptionListeners();
        };
    </script>
</head>
<body>
    <div class="container">
        <h1>Bus Driver Drowsiness Detection</h1>

        <div class="main-grid">
            <div class="left-panel">
                <div class="stats">
            <div class="stat-box">
                <h3>Status</h3>
                <h2 id="status" class="status-active">Active</h2>
            </div>
            <div class="stat-box">
                <h3>FPS</h3>
                <h2 id="fps">0</h2>
            </div>
            <div class="stat-box">
                <h3>Confidence</h3>
                <h2 id="confidence">0.00</h2>
            </div>
            <div class="stat-box">
                <h3>Blinks (60s)</h3>
                <h2 id="blinks_last_60s">0</h2>
            </div>
            <div class="stat-box">
                <h3>Check In</h3>
                <h2 id="fatigue_check">0s</h2>
            </div>
            <div class="stat-box">
                <h3>Yawns</h3>
                <h2 id="yawns">0</h2>
            </div>
            <div class="stat-box">
                <h3>Nods</h3>
                <h2 id="nods">0</h2>
            </div>
            <div class="stat-box">
                <h3>Yawn Phase</h3>
                <h2 id="yawn_phase">idle</h2>
            </div>
            <div class="stat-box">
                <h3>EAR</h3>
                <h2 id="ear">0.000</h2>
            </div>
            <div class="stat-box">
                <h3>EAR Thresh</h3>
                <h2 id="ear_threshold">0.000</h2>
            </div>
            <div class="stat-box">
                <h3>MAR</h3>
                <h2 id="mar">0.000</h2>
            </div>
            <div class="stat-box">
                <h3>MAR Thresh</h3>
                <h2 id="mar_threshold">0.000</h2>
            </div>
        </div>

        <div class="video-container">
            <img src="/video_feed" />
        </div>

        <div class="controls">
            <button onclick="recalibrate()">Recalibrate</button>
        </div>

        <div class="description-box" id="descriptionBox">
            <strong>Configuration Help:</strong>
            <p id="descriptionText">Hover over or click on a configuration field to see its description.</p>
        </div>
            </div>

            <div class="right-panel">
                <div class="state-section">
                    <h2>State Transition Formulas</h2>
                    <div class="state-formula">
                        <strong>ACTIVE → FATIGUED:</strong><br>
                        <div id="formula_active">Loading...</div>
                    </div>
                    <div class="state-formula">
                        <strong>FATIGUED → DROWSY:</strong><br>
                        <div id="formula_fatigued">Loading...</div>
                    </div>
                    <div class="state-formula">
                        <strong>Fatigue Trigger:</strong><br>
                        <div id="fatigue_triggers">Loading...</div>
                    </div>
                    <div class="counter-display">
                        <h3>Fatigue Events</h3>
                        <div class="counter-value" id="fatigue_events_count">0</div>
                    </div>
                </div>

                <div class="config-section">
                    <h2>Configuration</h2>
                    <div class="config-input">
                        <label>EAR % Threshold:</label>
                        <div class="input-with-arrows">
                            <button class="arrow-btn" onclick="adjustValue('EAR_PERCENTAGE_THRESHOLD', -1)">◀</button>
                            <input type="number" id="EAR_PERCENTAGE_THRESHOLD" step="0.01" min="0" max="1">
                            <button class="arrow-btn" onclick="adjustValue('EAR_PERCENTAGE_THRESHOLD', 1)">▶</button>
                        </div>
                    </div>
                    <div class="config-input">
                        <label>Eye Closure Seconds:</label>
                        <div class="input-with-arrows">
                            <button class="arrow-btn" onclick="adjustValue('EYE_CLOSURE_SECONDS_THRESHOLD', -1)">◀</button>
                            <input type="number" id="EYE_CLOSURE_SECONDS_THRESHOLD" step="0.1" min="0">
                            <button class="arrow-btn" onclick="adjustValue('EYE_CLOSURE_SECONDS_THRESHOLD', 1)">▶</button>
                        </div>
                    </div>
                    <div class="config-input">
                        <label>Yawn MAR Multiplier:</label>
                        <div class="input-with-arrows">
                            <button class="arrow-btn" onclick="adjustValue('YAWN_MAR_INCREASE_THRESH', -1)">◀</button>
                            <input type="number" id="YAWN_MAR_INCREASE_THRESH" step="0.01" min="1">
                            <button class="arrow-btn" onclick="adjustValue('YAWN_MAR_INCREASE_THRESH', 1)">▶</button>
                        </div>
                    </div>
                    <div class="config-input">
                        <label>Yawn Min Duration (s):</label>
                        <div class="input-with-arrows">
                            <button class="arrow-btn" onclick="adjustValue('YAWN_MIN_DURATION', -1)">◀</button>
                            <input type="number" id="YAWN_MIN_DURATION" step="0.1" min="0">
                            <button class="arrow-btn" onclick="adjustValue('YAWN_MIN_DURATION', 1)">▶</button>
                        </div>
                    </div>
                    <div class="config-input">
                        <label>Yawn Opening Rate:</label>
                        <div class="input-with-arrows">
                            <button class="arrow-btn" onclick="adjustValue('YAWN_OPENING_RATE_THRESH', -1)">◀</button>
                            <input type="number" id="YAWN_OPENING_RATE_THRESH" step="0.01" min="0">
                            <button class="arrow-btn" onclick="adjustValue('YAWN_OPENING_RATE_THRESH', 1)">▶</button>
                        </div>
                    </div>
                    <div class="config-input">
                        <label>Blink Consec Frames:</label>
                        <div class="input-with-arrows">
                            <button class="arrow-btn" onclick="adjustValue('BLINK_CONSEC_FRAMES', -1)">◀</button>
                            <input type="number" id="BLINK_CONSEC_FRAMES" step="1" min="1">
                            <button class="arrow-btn" onclick="adjustValue('BLINK_CONSEC_FRAMES', 1)">▶</button>
                        </div>
                    </div>
                    <div class="config-input">
                        <label>Blink Low Threshold:</label>
                        <div class="input-with-arrows">
                            <button class="arrow-btn" onclick="adjustValue('FATIGUE_LOW_BLINK_THRESHOLD', -1)">◀</button>
                            <input type="number" id="FATIGUE_LOW_BLINK_THRESHOLD" step="1" min="0">
                            <button class="arrow-btn" onclick="adjustValue('FATIGUE_LOW_BLINK_THRESHOLD', 1)">▶</button>
                        </div>
                    </div>
                    <div class="config-input">
                        <label>Blink High Threshold:</label>
                        <div class="input-with-arrows">
                            <button class="arrow-btn" onclick="adjustValue('FATIGUE_HIGH_BLINK_THRESHOLD', -1)">◀</button>
                            <input type="number" id="FATIGUE_HIGH_BLINK_THRESHOLD" step="1" min="0">
                            <button class="arrow-btn" onclick="adjustValue('FATIGUE_HIGH_BLINK_THRESHOLD', 1)">▶</button>
                        </div>
                    </div>
                    <div class="config-input">
                        <label>Drowsy Min Yawns:</label>
                        <div class="input-with-arrows">
                            <button class="arrow-btn" onclick="adjustValue('DROWSY_MIN_YAWNS', -1)">◀</button>
                            <input type="number" id="DROWSY_MIN_YAWNS" step="1" min="1">
                            <button class="arrow-btn" onclick="adjustValue('DROWSY_MIN_YAWNS', 1)">▶</button>
                        </div>
                    </div>
                    <div class="config-input">
                        <label>Min Confidence:</label>
                        <div class="input-with-arrows">
                            <button class="arrow-btn" onclick="adjustValue('MIN_CONFIDENCE', -1)">◀</button>
                            <input type="number" id="MIN_CONFIDENCE" step="0.01" min="0" max="1">
                            <button class="arrow-btn" onclick="adjustValue('MIN_CONFIDENCE', 1)">▶</button>
                        </div>
                    </div>
                    <div class="config-input">
                        <label>Fatigue Event Threshold:</label>
                        <div class="input-with-arrows">
                            <button class="arrow-btn" onclick="adjustValue('FATIGUE_EVENT_THRESHOLD', -1)">◀</button>
                            <input type="number" id="FATIGUE_EVENT_THRESHOLD" step="1" min="1">
                            <button class="arrow-btn" onclick="adjustValue('FATIGUE_EVENT_THRESHOLD', 1)">▶</button>
                        </div>
                    </div>
                    <div class="config-input">
                        <label>Drowsy Fatigue Events:</label>
                        <div class="input-with-arrows">
                            <button class="arrow-btn" onclick="adjustValue('DROWSY_MIN_FATIGUE_EVENTS', -1)">◀</button>
                            <input type="number" id="DROWSY_MIN_FATIGUE_EVENTS" step="1" min="1">
                            <button class="arrow-btn" onclick="adjustValue('DROWSY_MIN_FATIGUE_EVENTS', 1)">▶</button>
                        </div>
                    </div>
                    <div class="config-input">
                        <label>Auto Recalibrate (s):</label>
                        <div class="input-with-arrows">
                            <button class="arrow-btn" onclick="adjustValue('AUTO_RECALIBRATE_SECONDS', -1)">◀</button>
                            <input type="number" id="AUTO_RECALIBRATE_SECONDS" step="10" min="30">
                            <button class="arrow-btn" onclick="adjustValue('AUTO_RECALIBRATE_SECONDS', 1)">▶</button>
                        </div>
                    </div>
                    <div class="config-input">
                        <label>Face Lost Recal (s):</label>
                        <div class="input-with-arrows">
                            <button class="arrow-btn" onclick="adjustValue('FACE_LOST_RECALIBRATE_SECONDS', -1)">◀</button>
                            <input type="number" id="FACE_LOST_RECALIBRATE_SECONDS" step="0.5" min="0.5">
                            <button class="arrow-btn" onclick="adjustValue('FACE_LOST_RECALIBRATE_SECONDS', 1)">▶</button>
                        </div>
                    </div>
                    <div class="config-input">
                        <label>Fatigue Check Time (s):</label>
                        <div class="input-with-arrows">
                            <button class="arrow-btn" onclick="adjustValue('FATIGUE_BLINK_RATE_SECONDS', -1)">◀</button>
                            <input type="number" id="FATIGUE_BLINK_RATE_SECONDS" step="10" min="10">
                            <button class="arrow-btn" onclick="adjustValue('FATIGUE_BLINK_RATE_SECONDS', 1)">▶</button>
                        </div>
                    </div>
                    <div class="config-input">
                        <label>Face Active Time (s):</label>
                        <div class="input-with-arrows">
                            <button class="arrow-btn" onclick="adjustValue('FACE_ACTIVE_THRESHOLD_SECONDS', -1)">◀</button>
                            <input type="number" id="FACE_ACTIVE_THRESHOLD_SECONDS" step="0.5" min="0.5">
                            <button class="arrow-btn" onclick="adjustValue('FACE_ACTIVE_THRESHOLD_SECONDS', 1)">▶</button>
                        </div>
                    </div>
                    <button class="save-btn" onclick="saveConfig()">💾 Save Configuration</button>
                </div>
            </div>
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
                frame = frame_queue.get(timeout=0.5)
                # Higher JPEG quality for better image clarity
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            except:
                continue

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status')
def status():
    return jsonify(current_status)


@app.route('/recalibrate', methods=['POST'])
def recalibrate():
    global recalibrate_flag
    recalibrate_flag = True
    return jsonify({'status': 'started'})


@app.route('/get_config', methods=['GET'])
def get_config():
    global detector_instance
    if detector_instance:
        config = {
            "EAR_PERCENTAGE_THRESHOLD": detector_instance.EAR_PERCENTAGE_THRESHOLD,
            "EYE_CLOSURE_SECONDS_THRESHOLD": detector_instance.EYE_CLOSURE_SECONDS_THRESHOLD,
            "YAWN_MAR_INCREASE_THRESH": detector_instance.YAWN_MAR_INCREASE_THRESH,
            "YAWN_MIN_DURATION": detector_instance.YAWN_MIN_DURATION,
            "YAWN_OPENING_RATE_THRESH": detector_instance.YAWN_OPENING_RATE_THRESH,
            "BLINK_CONSEC_FRAMES": detector_instance.BLINK_CONSEC_FRAMES,
            "FATIGUE_LOW_BLINK_THRESHOLD": detector_instance.FATIGUE_LOW_BLINK_THRESHOLD,
            "FATIGUE_HIGH_BLINK_THRESHOLD": detector_instance.FATIGUE_HIGH_BLINK_THRESHOLD,
            "DROWSY_MIN_YAWNS": detector_instance.DROWSY_MIN_YAWNS,
            "MIN_CONFIDENCE": detector_instance.MIN_CONFIDENCE,
            "FATIGUE_EVENT_THRESHOLD": detector_instance.FATIGUE_EVENT_THRESHOLD,
            "DROWSY_MIN_FATIGUE_EVENTS": detector_instance.DROWSY_MIN_FATIGUE_EVENTS,
            "AUTO_RECALIBRATE_SECONDS": detector_instance.AUTO_RECALIBRATE_SECONDS,
            "FACE_LOST_RECALIBRATE_SECONDS": detector_instance.FACE_LOST_RECALIBRATE_SECONDS,
            "FATIGUE_BLINK_RATE_SECONDS": detector_instance.FATIGUE_BLINK_RATE_SECONDS,
            "FACE_ACTIVE_THRESHOLD_SECONDS": detector_instance.FACE_ACTIVE_THRESHOLD_SECONDS
        }
        return jsonify(config)
    return jsonify({"error": "Detector not initialized"}), 500


@app.route('/save_config', methods=['POST'])
def save_config():
    global detector_instance
    if detector_instance:
        data = request.json
        # Update detector instance
        for key, value in data.items():
            if hasattr(detector_instance, key):
                setattr(detector_instance, key, float(value))
        # Save to file
        if detector_instance.save_config():
            return jsonify({'status': 'success', 'message': 'Configuration saved'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to save configuration'}), 500
    return jsonify({'status': 'error', 'message': 'Detector not initialized'}), 500


if __name__ == "__main__":
    try:
        detector_instance = SimpleDrowsinessDetector()
        detection_thread = threading.Thread(target=detector_instance.run)
        detection_thread.daemon = True
        detection_thread.start()

        time.sleep(1)
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

    except KeyboardInterrupt:
        print("\n\n[INFO] System stopped")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback

        traceback.print_exc()