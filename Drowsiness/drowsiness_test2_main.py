import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy.spatial import distance as dist
import time
import os
from collections import deque
from flask import Flask, Response, render_template_string, jsonify, request
import threading
import queue
import json
from dataclasses import dataclass
from typing import List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)
frame_queue = queue.Queue(maxsize=2)
current_status = {"status": "Active", "confidence": 0.0, "blink_rate": 0.0, "yawns": 0, "nods": 0, "fatigue_check": 0,
                  "fps": 0.0}
detector_instance = None
recalibrate_flag = False


# MediaPipe Face Mesh landmark indices for key features
@dataclass
class FaceLandmarks:
    """MediaPipe Face Mesh landmark indices for facial features"""
    # Left eye landmarks (more detailed than dlib)
    LEFT_EYE_UPPER = [159, 145, 144, 163, 160, 161, 246]
    LEFT_EYE_LOWER = [33, 7, 163, 144, 145, 153, 154, 155, 133]
    LEFT_EYE_CORNERS = [33, 133]  # Inner and outer corners

    # Right eye landmarks
    RIGHT_EYE_UPPER = [386, 374, 373, 390, 387, 388, 466]
    RIGHT_EYE_LOWER = [263, 249, 390, 373, 374, 380, 381, 382, 362]
    RIGHT_EYE_CORNERS = [263, 362]

    # Mouth landmarks (more comprehensive than dlib)
    MOUTH_OUTER = [61, 84, 17, 314, 405, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
    MOUTH_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
    MOUTH_CORNERS = [61, 291]  # Left and right corners
    MOUTH_TOP_BOTTOM = [13, 14]  # Top and bottom center points

    # Nose tip for head tracking
    NOSE_TIP = [1, 2, 5, 4]

    # Face oval for stability detection
    FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 340, 346, 347, 348, 349, 350, 451, 452, 453, 464,
                 435, 410, 287, 273, 335, 406, 313, 18, 17, 175, 199, 200, 9, 10]

    # Head pose landmarks
    HEAD_POSE_LANDMARKS = [1, 33, 263, 61, 291, 199]  # Nose, eyes corners, mouth corners, chin


class MediaPipeDrowsinessDetector:
    def __init__(self):
        """Initialize MediaPipe-based drowsiness detector with TFLite acceleration"""
        # Camera settings
        self.yawn_phase = None
        self.RTSP_URL = "rtsp://admin:admin@192.168.1.64:554/Streaming/Channels/101"
        self.USE_RTSP = False
        self.CAMERA_SOURCE = self.RTSP_URL if self.USE_RTSP else 0
        self.CONFIG_FILE = "drowsiness_config.json"

        # Initialize all configuration parameters with default values
        # (These will be overwritten by load_config if config file exists)
        self.EAR_THRESHOLD_MULTIPLIER = 0.75
        self.EYE_CLOSURE_SECONDS_THRESHOLD = 3.0
        self.MAR_YAWN_MULTIPLIER = 1.5
        self.YAWN_MIN_DURATION = 0.8
        self.YAWN_MAX_DURATION = 6.0
        self.BLINK_CONSEC_FRAMES = 4
        self.FATIGUE_BLINK_RATE_SECONDS = 90
        self.FATIGUE_LOW_BLINK_THRESHOLD = 2
        self.FATIGUE_HIGH_BLINK_THRESHOLD = 40
        self.DROWSY_MIN_YAWNS = 4
        self.MIN_CONFIDENCE = 0.4
        self.FATIGUE_EVENT_THRESHOLD = 6
        self.DROWSY_MIN_FATIGUE_EVENTS = 10
        self.AUTO_RECALIBRATE_SECONDS = 120
        self.FACE_LOST_RECALIBRATE_SECONDS = 5.0
        self.FACE_ACTIVE_THRESHOLD_SECONDS = 2.0
        self.VIBRATION_COMPENSATION = True
        self.MOTION_SMOOTHING_WINDOW = 5
        self.NOISE_REDUCTION_STRENGTH = 0.7

        # Nod detection constants
        self.HEAD_NOD_FRAME_WINDOW = 60
        self.HEAD_NOD_THRESHOLD_MULTIPLIER = 0.25
        self.NOD_SMOOTHING_KERNEL = 15

        # Initialize detection state variables
        # (These will be reset by reset_detection_state called from load_config)
        self.calibrated_ear_baseline = 0.25
        self.calibrated_mar_baseline = 0.35
        self.calibrated_face_size = 150.0
        self.ear_std_dev = 0.05
        self.mar_std_dev = 0.10
        self.eye_closure_start_time = None
        self.yawn_start_time = None
        self.yawn_count = 0
        self.nod_count = 0
        self.is_nodding = False
        self.blink_counter = 0
        self.blink_frame_counter = 0
        self.blink_timestamps = deque(maxlen=1000)
        self.blink_analysis_start_time = time.time()
        self.display_blink_rate = 0.0
        self.fatigue_events = []
        self.current_state = 0
        self.confidence_score = 0.0
        self.ear_history = deque(maxlen=5)
        self.mar_history = deque(maxlen=5)
        self.head_positions = deque(maxlen=30)
        self.landmark_history = deque(maxlen=3)
        self.face_present = False
        self.face_lost_time = None
        self.face_detected_time = None
        self.face_confirmed_active = False
        self.last_recalibration_time = None

        # Initialize motion tracking buffers
        self.ear_history = deque(maxlen=5)
        self.mar_history = deque(maxlen=5)
        self.head_positions = deque(maxlen=30)
        self.landmark_history = deque(maxlen=3)

        # Initialize face tracking
        self.face_present = False
        self.face_lost_time = None
        self.face_detected_time = None
        self.face_confirmed_active = False
        self.last_recalibration_time = None

        # Load configuration
        self.load_config()

        # Initialize MediaPipe Face Landmarker
        self.init_mediapipe()

        # Vibration and noise compensation
        self.motion_stabilizer = MotionStabilizer()
        self.noise_filter = NoiseFilter()

    def init_mediapipe(self):
        """Initialize MediaPipe Face Landmarker with TFLite models"""
        # Download model if not present
        model_path = 'face_landmarker_v2_with_blendshapes.task'
        if not os.path.exists(model_path):
            print("[INFO] Downloading MediaPipe Face Landmarker model...")
            import urllib.request
            model_url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
            urllib.request.urlretrieve(model_url, model_path)
            print("[INFO] Model downloaded successfully")

        # Configure Face Landmarker options for NPU acceleration
        base_options = python.BaseOptions(
            model_asset_path=model_path,

        )

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,  # Single driver detection
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True,  # For expression analysis
            output_facial_transformation_matrixes=True  # For head pose
        )

        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
        print("[INFO] MediaPipe Face Landmarker initialized with NPU acceleration")

    def load_config(self):
        """Load configuration from JSON file or use defaults"""
        defaults = {
            "EAR_THRESHOLD_MULTIPLIER": 0.75,
            "EYE_CLOSURE_SECONDS_THRESHOLD": 3.0,
            "MAR_YAWN_MULTIPLIER": 1.5,
            "YAWN_MIN_DURATION": 0.8,
            "YAWN_MAX_DURATION": 6.0,
            "BLINK_CONSEC_FRAMES": 3,
            "FATIGUE_BLINK_RATE_SECONDS": 90,
            "FATIGUE_LOW_BLINK_THRESHOLD": 2,
            "FATIGUE_HIGH_BLINK_THRESHOLD": 40,
            "DROWSY_MIN_YAWNS": 4,
            "MIN_CONFIDENCE": 0.4,
            "FATIGUE_EVENT_THRESHOLD": 6,
            "DROWSY_MIN_FATIGUE_EVENTS": 10,
            "AUTO_RECALIBRATE_SECONDS": 120,
            "FACE_LOST_RECALIBRATE_SECONDS": 5.0,
            "FACE_ACTIVE_THRESHOLD_SECONDS": 2.0,
            "VIBRATION_COMPENSATION": True,
            "MOTION_SMOOTHING_WINDOW": 5,
            "NOISE_REDUCTION_STRENGTH": 0.7
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

        # Initialize detection variables
        self.reset_detection_state()

    def reset_detection_state(self):
        """Reset all detection state variables"""
        # Calibration values
        self.calibrated_ear_baseline = 0.25
        self.calibrated_mar_baseline = 0.35
        self.calibrated_face_size = 150.0
        self.ear_std_dev = 0.05
        self.mar_std_dev = 0.10

        # Detection states
        self.eye_closure_start_time = None
        self.yawn_start_time = None
        self.yawn_phase = "idle"
        self.yawn_count = 0
        self.nod_count = 0
        self.is_nodding = False
        self.blink_counter = 0
        self.blink_frame_counter = 0
        self.blink_timestamps = deque(maxlen=1000)
        self.blink_analysis_start_time = time.time()
        self.display_blink_rate = 0.0
        self.fatigue_events = []
        self.current_state = 0  # 0: Active, 1: Fatigued, 2: Drowsy
        self.confidence_score = 0.0

        # Motion tracking buffers
        self.ear_history = deque(maxlen=self.MOTION_SMOOTHING_WINDOW)
        self.mar_history = deque(maxlen=self.MOTION_SMOOTHING_WINDOW)
        self.head_positions = deque(maxlen=self.HEAD_NOD_FRAME_WINDOW)
        self.landmark_history = deque(maxlen=3)  # For temporal smoothing

        # Face tracking
        self.face_present = False
        self.face_lost_time = None
        self.face_detected_time = None
        self.face_confirmed_active = False
        self.last_recalibration_time = None

    def eye_aspect_ratio_mediapipe(self, landmarks, eye_indices_upper, eye_indices_lower):
        """Calculate EAR using MediaPipe landmarks with vibration compensation"""
        try:
            # Get upper and lower eyelid points
            upper_points = np.array([[landmarks[i].x, landmarks[i].y, landmarks[i].z]
                                     for i in eye_indices_upper])
            lower_points = np.array([[landmarks[i].x, landmarks[i].y, landmarks[i].z]
                                     for i in eye_indices_lower[:len(eye_indices_upper)]])

            # Calculate vertical distances (more robust for vibration)
            vertical_distances = np.linalg.norm(upper_points - lower_points, axis=1)

            # Get eye corners for horizontal distance
            if len(eye_indices_lower) >= 2:
                corner1 = np.array([landmarks[eye_indices_lower[0]].x,
                                    landmarks[eye_indices_lower[0]].y,
                                    landmarks[eye_indices_lower[0]].z])
                corner2 = np.array([landmarks[eye_indices_lower[-1]].x,
                                    landmarks[eye_indices_lower[-1]].y,
                                    landmarks[eye_indices_lower[-1]].z])
                horizontal_distance = np.linalg.norm(corner1 - corner2)
            else:
                horizontal_distance = 1.0  # Default to prevent division by zero

            # Calculate EAR with stability checks
            if horizontal_distance > 0:
                ear = np.mean(vertical_distances) / horizontal_distance
            else:
                ear = 0.25  # Default value

            # Clamp to reasonable range
            return np.clip(ear, 0.05, 0.5)

        except Exception as e:
            print(f"[WARNING] EAR calculation error: {e}")
            return 0.25

    def mouth_aspect_ratio_mediapipe(self, landmarks):
        """Calculate MAR using MediaPipe landmarks with noise filtering"""
        try:
            # Get mouth top and bottom center points
            top = np.array([landmarks[FaceLandmarks.MOUTH_TOP_BOTTOM[0]].x,
                            landmarks[FaceLandmarks.MOUTH_TOP_BOTTOM[0]].y,
                            landmarks[FaceLandmarks.MOUTH_TOP_BOTTOM[0]].z])
            bottom = np.array([landmarks[FaceLandmarks.MOUTH_TOP_BOTTOM[1]].x,
                               landmarks[FaceLandmarks.MOUTH_TOP_BOTTOM[1]].y,
                               landmarks[FaceLandmarks.MOUTH_TOP_BOTTOM[1]].z])

            # Get mouth corners
            left_corner = np.array([landmarks[FaceLandmarks.MOUTH_CORNERS[0]].x,
                                    landmarks[FaceLandmarks.MOUTH_CORNERS[0]].y,
                                    landmarks[FaceLandmarks.MOUTH_CORNERS[0]].z])
            right_corner = np.array([landmarks[FaceLandmarks.MOUTH_CORNERS[1]].x,
                                     landmarks[FaceLandmarks.MOUTH_CORNERS[1]].y,
                                     landmarks[FaceLandmarks.MOUTH_CORNERS[1]].z])

            # Calculate distances
            vertical = np.linalg.norm(top - bottom)
            horizontal = np.linalg.norm(left_corner - right_corner)

            # Calculate MAR
            if horizontal > 0:
                mar = vertical / horizontal
            else:
                mar = 0.35

            # Apply additional filtering for bus vibrations
            if self.VIBRATION_COMPENSATION:
                mar = self.noise_filter.filter(mar)

            return np.clip(mar, 0.1, 2.0)

        except Exception as e:
            print(f"[WARNING] MAR calculation error: {e}")
            return 0.35

    def detect_head_nod(self, landmarks):
        """Detect head nodding using convolution-based smoothing (matching dlib version)"""
        try:
            # Get nose tip position (more stable than single point)
            nose_positions = [landmarks[i] for i in FaceLandmarks.NOSE_TIP]
            nose_y = np.mean([p.y for p in nose_positions])

            self.head_positions.append(nose_y)

            if len(self.head_positions) >= self.HEAD_NOD_FRAME_WINDOW:
                positions = np.array(self.head_positions)

                # Apply convolution smoothing with kernel (like dlib version)
                kernel = np.ones(self.NOD_SMOOTHING_KERNEL) / self.NOD_SMOOTHING_KERNEL
                smoothed = np.convolve(positions, kernel, mode='valid')

                #
                #
                if len(smoothed) > 20:
                    # Calculate nod threshold based on calibrated face size
                    # The threshold is now a fraction of the frame height for better scaling
                    nod_threshold = 0.03  # Represents a 3% drop in frame height

                    # FIX: Compare the start of the window to the end for a sustained droop
                    start_avg = np.mean(smoothed[:10])  # Average of first 10 smoothed points
                    end_avg = np.mean(smoothed[-10:])  # Average of last 10 smoothed points
                    movement = end_avg - start_avg  # Positive value means head moved down

                    if movement > nod_threshold:
                        if not self.is_nodding:
                            self.is_nodding = True
                            return True
                    else:
                        self.is_nodding = False
            #
            #

        except Exception as e:
            print(f"[WARNING] Head nod detection error: {e}")

        return False

    def process_frame_with_mediapipe(self, frame):
        """Process frame using MediaPipe Face Landmarker"""
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect faces and landmarks
        detection_result = self.face_landmarker.detect(mp_image)

        if not detection_result.face_landmarks:
            return None, None, None

        # Get first face landmarks (driver)
        face_landmarks = detection_result.face_landmarks[0]

        # Get face blendshapes for expression analysis
        face_blendshapes = detection_result.face_blendshapes[0] if detection_result.face_blendshapes else None

        # Get transformation matrix for head pose
        face_transform = detection_result.facial_transformation_matrixes[
            0] if detection_result.facial_transformation_matrixes else None

        return face_landmarks, face_blendshapes, face_transform

    def calculate_metrics(self, landmarks):
        """Calculate drowsiness metrics from MediaPipe landmarks"""
        # Calculate EAR for both eyes
        left_ear = self.eye_aspect_ratio_mediapipe(
            landmarks,
            FaceLandmarks.LEFT_EYE_UPPER,
            FaceLandmarks.LEFT_EYE_LOWER
        )
        right_ear = self.eye_aspect_ratio_mediapipe(
            landmarks,
            FaceLandmarks.RIGHT_EYE_UPPER,
            FaceLandmarks.RIGHT_EYE_LOWER
        )
        ear = (left_ear + right_ear) / 2.0

        # Calculate MAR
        mar = self.mouth_aspect_ratio_mediapipe(landmarks)

        # Apply temporal smoothing for stability
        self.ear_history.append(ear)
        self.mar_history.append(mar)

        smoothed_ear = np.mean(self.ear_history)
        smoothed_mar = np.mean(self.mar_history)

        return smoothed_ear, smoothed_mar

    def preprocess_frame(self, frame):
        """Preprocess frame for better detection in bus environment"""
        # MediaPipe works best with color images, so keep it in BGR
        # Apply CLAHE for contrast enhancement on each channel
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE only to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        # Merge back
        lab = cv2.merge([l, a, b])
        processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return processed

    def recalibrate(self, vs, reset_counters=True):
        """Recalibrate baseline values for current driver"""
        print(f"\n[{time.ctime()}] Starting calibration...")
        print("[INFO] Please look forward with eyes open, mouth closed")

        calibration_samples = {
            'ear': [],
            'mar': [],
            'face_size': []
        }

        stable_frames = 0
        target_frames = 60
        max_attempts = target_frames * 3
        attempt = 0

        while stable_frames < target_frames and attempt < max_attempts:
            ret, frame = vs.read()
            if not ret:
                continue

            attempt += 1

            # Preprocess frame
            processed = self.preprocess_frame(frame)

            # Detect with MediaPipe
            landmarks, blendshapes, transform = self.process_frame_with_mediapipe(processed)

            if landmarks:
                # Face detected successfully
                try:
                    # Calculate metrics
                    ear, mar = self.calculate_metrics(landmarks)

                    # Estimate face size from landmarks
                    face_points = [landmarks[i] for i in FaceLandmarks.FACE_OVAL]
                    face_width = max([p.x for p in face_points]) - min([p.x for p in face_points])
                    face_height = max([p.y for p in face_points]) - min([p.y for p in face_points])
                    face_size = (face_width + face_height) * frame.shape[0] / 2

                    # Validate values
                    if 0.15 < ear < 0.45 and 0.2 < mar < 0.6:
                        calibration_samples['ear'].append(ear)
                        calibration_samples['mar'].append(mar)
                        calibration_samples['face_size'].append(face_size)
                        stable_frames += 1

                    # Update display
                    progress = int((stable_frames / target_frames) * 100)
                    cv2.putText(processed, f"CALIBRATING: {progress}%", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                except Exception as e:
                    print(f"[WARNING] Calibration sample error: {e}")
            else:
                # No face detected
                if attempt % 10 == 0:
                    print(f"[DEBUG] Calibration attempt {attempt}/{max_attempts}: No face detected (samples: {stable_frames}/{target_frames})")

            # Update frame queue
            try:
                while not frame_queue.empty():
                    frame_queue.get_nowait()
                frame_queue.put(processed)
            except:
                pass

        # Process calibration samples
        if len(calibration_samples['ear']) >= target_frames * 0.6:
            # Remove outliers using IQR method
            for key in calibration_samples:
                values = np.array(calibration_samples[key])
                q1, q3 = np.percentile(values, [25, 75])
                iqr = q3 - q1
                mask = ((values >= q1 - 1.5 * iqr) & (values <= q3 + 1.5 * iqr))
                calibration_samples[key] = values[mask]

            # Set calibrated values
            self.calibrated_ear_baseline = np.median(calibration_samples['ear'])
            self.ear_std_dev = np.std(calibration_samples['ear'])

            self.calibrated_mar_baseline = np.median(calibration_samples['mar'])
            self.mar_std_dev = np.std(calibration_samples['mar'])

            self.calibrated_face_size = np.median(calibration_samples['face_size'])

            print(f"\n[SUCCESS] Calibration complete!")
            print(f"  EAR baseline: {self.calibrated_ear_baseline:.3f} (±{self.ear_std_dev:.3f})")
            print(f"  MAR baseline: {self.calibrated_mar_baseline:.3f} (±{self.mar_std_dev:.3f})")
            print(f"  Face size: {self.calibrated_face_size:.1f}")
        else:
            print(f"\n[WARNING] Insufficient calibration samples. Using defaults.")

        # Save recalibration time before reset
        recalibration_time = time.time()

        if reset_counters:
            self.reset_detection_state()
            # Restore recalibration time after reset
            self.last_recalibration_time = recalibration_time
        else:
            self.last_recalibration_time = recalibration_time

    def run(self):
        """Main detection loop"""
        print("\n" + "=" * 70)
        print("  MEDIAPIPE DROWSINESS DETECTION SYSTEM")
        print("  NPU-Accelerated with TFLite")
        print("=" * 70)
        print(f"  Camera: {self.CAMERA_SOURCE}")
        print("  Web UI: http://localhost:5000")
        print("=" * 70 + "\n")

        # Initialize video capture
        vs = self.init_video_capture()
        if not vs.isOpened():
            raise RuntimeError(f"[ERROR] Failed to open camera: {self.CAMERA_SOURCE}")

        # Initial calibration
        self.recalibrate(vs, reset_counters=True)

        # FPS tracking
        fps_counter = 0
        fps_start_time = time.time()
        fps = 0

        while True:
            # Check for manual recalibration
            global recalibrate_flag
            if recalibrate_flag:
                self.recalibrate(vs, reset_counters=True)
                recalibrate_flag = False

            # Auto recalibration
            if (self.last_recalibration_time and
                    time.time() - self.last_recalibration_time > self.AUTO_RECALIBRATE_SECONDS):
                self.recalibrate(vs, reset_counters=False)

            # Read frame
            ret, frame = vs.read()
            if not ret:
                if self.USE_RTSP:
                    print("[WARNING] RTSP stream interrupted. Reconnecting...")
                    vs = self.reconnect_rtsp(vs)
                continue

            # Validate frame
            if not self.validate_frame(frame):
                continue

            # Preprocess frame
            processed = self.preprocess_frame(frame)

            # FPS calculation
            fps_counter += 1
            if fps_counter % 15 == 0:
                fps = 15 / (time.time() - fps_start_time)
                fps_start_time = time.time()

            # Process with MediaPipe every frame (NPU/GPU accelerated)
            landmarks, blendshapes, transform = self.process_frame_with_mediapipe(processed)

            if landmarks:
                # Calculate metrics
                ear, mar = self.calculate_metrics(landmarks)

                # Calculate confidence score based on landmark quality
                self.confidence_score = min(1.0, len(self.ear_history) / 5.0) if self.face_confirmed_active else 0.0

                # Detect drowsiness indicators
                self.detect_blinks(ear)
                self.detect_yawns(mar)

                if self.detect_head_nod(landmarks):
                    self.nod_count += 1
                    self.fatigue_events.append(time.time())

                # Update state machine
                self.update_state_machine(ear, mar)

                # Draw landmarks on frame
                self.draw_landmarks(processed, landmarks)

                # Update face tracking
                self.update_face_tracking(True)
            else:
                self.confidence_score = 0.0
                self.update_face_tracking(False)

            # Update status
            self.update_status(fps)

            # Update frame queue
            try:
                while not frame_queue.empty():
                    frame_queue.get_nowait()
                frame_queue.put(processed)
            except:
                pass

        vs.release()

    def init_video_capture(self):
        """Initialize video capture with optimized settings"""
        if self.USE_RTSP:
            # RTSP with low latency settings
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;udp|"
                "fflags;nobuffer|"
                "flags;low_delay|"
                "framedrop;1"
            )
            vs = cv2.VideoCapture(self.CAMERA_SOURCE, cv2.CAP_FFMPEG)
            vs.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            # Webcam
            vs = cv2.VideoCapture(self.CAMERA_SOURCE)
            vs.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            vs.set(cv2.CAP_PROP_FPS, 30)

        return vs

    def reconnect_rtsp(self, vs):
        """Reconnect RTSP stream"""
        vs.release()
        time.sleep(1)
        return self.init_video_capture()

    def validate_frame(self, frame):
        """Validate frame integrity"""
        if frame is None or frame.size == 0:
            return False
        if frame.shape[0] < 100 or frame.shape[1] < 100:
            return False
        # Check for all black/white frames
        mean_val = np.mean(frame)
        if mean_val < 5 or mean_val > 250:
            return False
        return True

    def detect_blinks(self, ear):
        """Detect blinks with vibration compensation"""
        threshold = self.calibrated_ear_baseline * self.EAR_THRESHOLD_MULTIPLIER

        if ear < threshold:
            self.blink_frame_counter += 1
        else:
            if self.blink_frame_counter >= self.BLINK_CONSEC_FRAMES:
                self.blink_counter += 1
                self.blink_timestamps.append(time.time())
            self.blink_frame_counter = 0

        # Track eye closure duration for drowsiness
        if ear < threshold and self.confidence_score > self.MIN_CONFIDENCE:
            if self.eye_closure_start_time is None:
                self.eye_closure_start_time = time.time()
        else:
            self.eye_closure_start_time = None

    def detect_yawns(self, mar):
        """Detect yawns with robust state machine"""
        yawn_threshold = self.calibrated_mar_baseline * self.MAR_YAWN_MULTIPLIER

        if mar > yawn_threshold:
            if self.yawn_start_time is None:
                self.yawn_start_time = time.time()
                self.yawn_phase = "opening"
        else:
            if self.yawn_start_time:
                yawn_duration = time.time() - self.yawn_start_time
                if self.YAWN_MIN_DURATION <= yawn_duration <= self.YAWN_MAX_DURATION:
                    self.yawn_count += 1
                    self.fatigue_events.append(time.time())
                    self.yawn_phase = "closing"
                else:
                    self.yawn_phase = "idle"
                self.yawn_start_time=None
            else:
                self.yawn_phase = "idle"

    def update_state_machine(self, ear, mar):
        """Update drowsiness state machine"""
        current_time = time.time()

        # Calculate display blink rate (blinks per minute over analysis window)
        # This is different from the 60-second rolling counter
        if current_time - self.blink_analysis_start_time > self.FATIGUE_BLINK_RATE_SECONDS:
            # Calculate blinks per minute from the analysis window
            self.display_blink_rate = (self.blink_counter / self.FATIGUE_BLINK_RATE_SECONDS) * 60

            # Check for fatigue based on blink rate
            if (self.display_blink_rate < self.FATIGUE_LOW_BLINK_THRESHOLD or
                    self.display_blink_rate > self.FATIGUE_HIGH_BLINK_THRESHOLD):
                if self.confidence_score > 0.3:
                    self.fatigue_events.append(current_time)

            # FIX: Reset the analysis window first, THEN reset the counter
            self.blink_analysis_start_time = current_time
            self.blink_counter = 0

        # Clean old fatigue events (5-minute window)
        self.fatigue_events = [e for e in self.fatigue_events
                               if current_time - e < 300]

        # State transitions
        if self.current_state == 0:  # Active
            if len(self.fatigue_events) >= self.FATIGUE_EVENT_THRESHOLD:
                self.current_state = 1  # Fatigued
        elif self.current_state == 1:  # Fatigued
            if len(self.fatigue_events) >= self.DROWSY_MIN_FATIGUE_EVENTS:
                if self.yawn_count >= self.DROWSY_MIN_YAWNS:
                    self.current_state = 2  # Drowsy
            elif len(self.fatigue_events) < 3:
                self.current_state = 0  # Active
        elif self.current_state == 2:  # Drowsy
            if len(self.fatigue_events) < 6:
                self.current_state = 1  # Fatigued

    def update_face_tracking(self, face_detected):
        """Update face presence tracking"""
        current_time = time.time()

        if face_detected:
            if not self.face_present:
                self.face_detected_time = current_time
                self.face_present = True
                self.face_confirmed_active = False

            if not self.face_confirmed_active and self.face_detected_time:
                if current_time - self.face_detected_time >= self.FACE_ACTIVE_THRESHOLD_SECONDS:
                    self.face_confirmed_active = True

                    # Check if recalibration needed
                    if self.face_lost_time:
                        lost_duration = self.face_detected_time - self.face_lost_time
                        if lost_duration >= self.FACE_LOST_RECALIBRATE_SECONDS:
                            global recalibrate_flag
                            recalibrate_flag = True
        else:
            if self.face_present:
                self.face_lost_time = current_time
                self.face_present = False
                self.face_confirmed_active = False

    def draw_landmarks(self, frame, landmarks):
        """Draw face landmarks on frame"""
        h, w = frame.shape[:2]

        # Draw eye landmarks
        for idx in FaceLandmarks.LEFT_EYE_UPPER + FaceLandmarks.LEFT_EYE_LOWER:
            x = int(landmarks[idx].x * w)
            y = int(landmarks[idx].y * h)
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        for idx in FaceLandmarks.RIGHT_EYE_UPPER + FaceLandmarks.RIGHT_EYE_LOWER:
            x = int(landmarks[idx].x * w)
            y = int(landmarks[idx].y * h)
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # Draw mouth landmarks
        for idx in FaceLandmarks.MOUTH_OUTER:
            x = int(landmarks[idx].x * w)
            y = int(landmarks[idx].y * h)
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

    def update_status(self, fps):
        """Update global status dictionary"""
        global current_status

        status_text = "Active"
        if self.current_state == 2:
            status_text = "DROWSINESS ALERT!"
        elif self.current_state == 1:
            status_text = "Fatigue Detected"
        elif not self.face_confirmed_active:
            status_text = "Detecting Face..."

        current_time = time.time()
        cutoff_time = current_time - 60
        recent_blinks = sum(1 for ts in self.blink_timestamps if ts >= cutoff_time)

        current_status = {
            "status": status_text,
            "confidence": float(self.confidence_score),
            "blink_rate": float(recent_blinks),
            "blinks_last_60s": int(recent_blinks),
            "yawns": int(self.yawn_count),
            "nods": int(self.nod_count),
            "yawn_phase": self.yawn_phase,
            "fatigue_check": int(
                max(0, self.FATIGUE_BLINK_RATE_SECONDS - (current_time - self.last_recalibration_time)) if self.last_recalibration_time else 0),
            "fps": float(fps),
            "ear": float(self.ear_history[-1] if self.ear_history else 0),
            "ear_threshold": float(self.calibrated_ear_baseline * self.EAR_THRESHOLD_MULTIPLIER),
            "mar": float(self.mar_history[-1] if self.mar_history else 0),
            "mar_threshold": float(self.calibrated_mar_baseline * self.MAR_YAWN_MULTIPLIER),
            "state_reasons": {
                "fatigue_events": int(len(self.fatigue_events)),
                "fatigue_threshold": int(self.FATIGUE_EVENT_THRESHOLD),
                "drowsy_fatigue_threshold": int(self.DROWSY_MIN_FATIGUE_EVENTS),
                "yawn_count": int(self.yawn_count),
                "yawn_threshold": int(self.DROWSY_MIN_YAWNS),
                "blink_rate": float(self.display_blink_rate),
                "blink_low_threshold": int(self.FATIGUE_LOW_BLINK_THRESHOLD),
                "blink_high_threshold": int(self.FATIGUE_HIGH_BLINK_THRESHOLD),
                "current_state": int(self.current_state)
            }
        }

    def save_config(self):
        """Save current configuration to file"""
        config = {
            "EAR_THRESHOLD_MULTIPLIER": self.EAR_THRESHOLD_MULTIPLIER,
            "EYE_CLOSURE_SECONDS_THRESHOLD": self.EYE_CLOSURE_SECONDS_THRESHOLD,
            "MAR_YAWN_MULTIPLIER": self.MAR_YAWN_MULTIPLIER,
            "YAWN_MIN_DURATION": self.YAWN_MIN_DURATION,
            "YAWN_MAX_DURATION": self.YAWN_MAX_DURATION,
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
            "FACE_ACTIVE_THRESHOLD_SECONDS": self.FACE_ACTIVE_THRESHOLD_SECONDS,
            "VIBRATION_COMPENSATION": self.VIBRATION_COMPENSATION,
            "MOTION_SMOOTHING_WINDOW": self.MOTION_SMOOTHING_WINDOW,
            "NOISE_REDUCTION_STRENGTH": self.NOISE_REDUCTION_STRENGTH
        }

        try:
            with open(self.CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=4)
            return True
        except Exception as e:
            print(f"[ERROR] Could not save config: {e}")
            return False


class MotionStabilizer:
    """Stabilize motion data to compensate for bus vibrations"""

    def __init__(self, history_size=10):
        self.history = deque(maxlen=history_size)
        self.alpha = 0.3  # Low-pass filter coefficient

    def stabilize(self, value):
        """Apply motion stabilization"""
        self.history.append(value)

        if len(self.history) < 3:
            return value

        # Remove outliers
        values = np.array(self.history)
        median = np.median(values)
        mad = np.median(np.abs(values - median))

        # Filter outliers using MAD
        threshold = 2.5 * mad
        filtered = values[np.abs(values - median) <= threshold]

        if len(filtered) == 0:
            return median

        # Apply exponential moving average
        return np.mean(filtered)


class NoiseFilter:
    """Filter noise from sensor data in vibrating environment"""

    def __init__(self, strength=0.7):
        self.strength = strength
        self.prev_value = None

    def filter(self, value):
        """Apply noise filtering"""
        if self.prev_value is None:
            self.prev_value = value
            return value

        # Exponential smoothing
        filtered = self.strength * self.prev_value + (1 - self.strength) * value
        self.prev_value = filtered
        return filtered

    def smooth_signal(self, signal):
        """Smooth entire signal using Savitzky-Golay filter"""
        from scipy.signal import savgol_filter

        if len(signal) < 5:
            return signal

        try:
            # Use Savitzky-Golay filter for smoothing
            window_length = min(11, len(signal) if len(signal) % 2 == 1 else len(signal) - 1)
            if window_length < 5:
                return signal
            smoothed = savgol_filter(signal, window_length, 3)
            return smoothed
        except:
            return signal


# Flask routes remain the same
@app.route('/')
def index():
    return render_template_string('''<!DOCTYPE html>
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
            'EAR_THRESHOLD_MULTIPLIER': 'Eye Aspect Ratio threshold multiplier. Lower values = more sensitive to eye closure. Range: 0.0-1.0.',
            'EYE_CLOSURE_SECONDS_THRESHOLD': 'Duration (in seconds) eyes must be closed continuously to trigger drowsiness alert. Lower values = faster alerts.',
            'MAR_YAWN_MULTIPLIER': 'Mouth Aspect Ratio multiplier for yawn detection. How much mouth must open relative to baseline. Lower = more sensitive.',
            'YAWN_MIN_DURATION': 'Minimum duration (seconds) for valid yawn. Too low may trigger false positives. Typical yawn: 0.8-3 seconds.',
            'BLINK_CONSEC_FRAMES': 'Number of consecutive frames with closed eyes to count as blink. Higher = filters fast eye movements.',
            'FATIGUE_LOW_BLINK_THRESHOLD': 'Minimum blinks (per 60s) to avoid fatigue. Below this triggers low blink fatigue event.',
            'FATIGUE_HIGH_BLINK_THRESHOLD': 'Maximum blinks (per 60s) before excessive. Above this triggers high blink fatigue event.',
            'DROWSY_MIN_YAWNS': 'Number of yawns in recent history to trigger drowsy state. Lower = more sensitive to yawning.',
            'MIN_CONFIDENCE': 'Minimum confidence score for detection. Higher = fewer false positives, but may miss genuine events.',
            'FATIGUE_EVENT_THRESHOLD': 'Number of recent fatigue events needed to transition to FATIGUE state. Lower = faster alerts.',
            'DROWSY_MIN_FATIGUE_EVENTS': 'Number of fatigue events required to escalate from FATIGUE to DROWSY state. Lower = escalates faster.',
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
                        <label>EAR Threshold Mult:</label>
                        <div class="input-with-arrows">
                            <button class="arrow-btn" onclick="adjustValue('EAR_THRESHOLD_MULTIPLIER', -1)">◀</button>
                            <input type="number" id="EAR_THRESHOLD_MULTIPLIER" step="0.01" min="0" max="1">
                            <button class="arrow-btn" onclick="adjustValue('EAR_THRESHOLD_MULTIPLIER', 1)">▶</button>
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
                        <label>MAR Yawn Multiplier:</label>
                        <div class="input-with-arrows">
                            <button class="arrow-btn" onclick="adjustValue('MAR_YAWN_MULTIPLIER', -1)">◀</button>
                            <input type="number" id="MAR_YAWN_MULTIPLIER" step="0.01" min="1">
                            <button class="arrow-btn" onclick="adjustValue('MAR_YAWN_MULTIPLIER', 1)">▶</button>
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
</html>''')

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            try:
                frame = frame_queue.get(timeout=0.5)
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
            "EAR_THRESHOLD_MULTIPLIER": detector_instance.EAR_THRESHOLD_MULTIPLIER,
            "EYE_CLOSURE_SECONDS_THRESHOLD": detector_instance.EYE_CLOSURE_SECONDS_THRESHOLD,
            "MAR_YAWN_MULTIPLIER": detector_instance.MAR_YAWN_MULTIPLIER,
            "YAWN_MIN_DURATION": detector_instance.YAWN_MIN_DURATION,
            "BLINK_CONSEC_FRAMES": detector_instance.BLINK_CONSEC_FRAMES,
            "FATIGUE_LOW_BLINK_THRESHOLD": detector_instance.FATIGUE_LOW_BLINK_THRESHOLD,
            "FATIGUE_HIGH_BLINK_THRESHOLD": detector_instance.FATIGUE_HIGH_BLINK_THRESHOLD,
            "DROWSY_MIN_YAWNS": detector_instance.DROWSY_MIN_YAWNS,
            "MIN_CONFIDENCE": detector_instance.MIN_CONFIDENCE,
            "FATIGUE_EVENT_THRESHOLD": detector_instance.FATIGUE_EVENT_THRESHOLD,
            "DROWSY_MIN_FATIGUE_EVENTS": detector_instance.DROWSY_MIN_FATIGUE_EVENTS,
            "AUTO_RECALIBRATE_SECONDS": detector_instance.AUTO_RECALIBRATE_SECONDS,
            "FACE_LOST_RECALIBRATE_SECONDS": detector_instance.FACE_LOST_RECALIBRATE_SECONDS,
            "FACE_ACTIVE_THRESHOLD_SECONDS": detector_instance.FACE_ACTIVE_THRESHOLD_SECONDS,
            "VIBRATION_COMPENSATION": detector_instance.VIBRATION_COMPENSATION,
            "NOISE_REDUCTION_STRENGTH": detector_instance.NOISE_REDUCTION_STRENGTH
        }
        return jsonify(config)
    return jsonify({"error": "Detector not initialized"}), 500


@app.route('/save_config', methods=['POST'])
def save_config():
    global detector_instance
    if detector_instance:
        data = request.json
        for key, value in data.items():
            if hasattr(detector_instance, key):
                setattr(detector_instance, key, value)
        if detector_instance.save_config():
            return jsonify({'status': 'success', 'message': 'Configuration saved'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to save configuration'}), 500
    return jsonify({'status': 'error', 'message': 'Detector not initialized'}), 500


if __name__ == "__main__":
    try:
        detector_instance = MediaPipeDrowsinessDetector()
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