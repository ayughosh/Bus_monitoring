# Robust Drowsiness Detector with Web Streaming
# Modified to display on webpage instead of CV2 window

import cv2
import dlib
from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import os
from collections import deque
import time
import threading
import queue
from flask import Flask, Response, render_template_string, jsonify


# Constants - ADJUSTED FOR MOVEMENT
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 60
YAWN_MAR_THRESH = 0.6
MOUTH_CHANGE_THRESH = 0.05

# Yawn Detection Parameters
YAWN_MIN_DURATION = 1.0  # Minimum duration in seconds for valid yawn
YAWN_MAX_DURATION = 6.0  # Maximum duration in seconds for valid yawn
YAWN_OPENING_RATE_THRESH = 0.05  # Minimum MAR increase rate per frame
YAWN_MAR_INCREASE_THRESH = 1.5  # Yawn MAR must be 1.5x baseline
YAWN_COOLDOWN_FRAMES = 30  # Frames to wait before detecting another yawn

CALIBRATION_FRAMES = 100
MIN_CONFIDENCE_THRESHOLD = 0.60
SMOOTH_WINDOW_SIZE = 7
OUTLIER_FACTOR = 2.5
MIN_FACE_SIZE = 60
MAX_FACE_MOVEMENT = 0.6
INTERPOLATION_WEIGHT = 0.7

# State variables
EYE_CLOSED_COUNTER = 0
DROWSY_ALERT = False
confidence_score = 0.0

# Calibration variables
eye_calibrated = False
mouth_calibrated = False
cal_eye_ratios = []
cal_mouth_ratios = []
calibrated_eye_threshold = 0.0
calibrated_mouth_baseline = 0.0
calibrated_mouth_threshold = 0.0
eye_std_dev = 0.0
mouth_std_dev = 0.0

# Smoothing buffers
ear_history = deque(maxlen=SMOOTH_WINDOW_SIZE)
mar_history = deque(maxlen=SMOOTH_WINDOW_SIZE)
last_face_width = None
face_lost_counter = 0
last_ear = 0.30
last_mar = 0.30
last_valid_frame_time = time.time()

# Yawn detection state
yawn_state = "IDLE"  # States: IDLE, OPENING, PEAK, CLOSING
yawn_start_time = None
yawn_peak_mar = 0.0
yawn_frame_count = 0
yawn_cooldown_counter = 0
previous_mar = 0.0
yawn_detected_count = 0
mar_rate_history = deque(maxlen=5)

# Web streaming variables
frame_queue = queue.Queue(maxsize=10)
latest_frame = None
current_status = {"status": "Active", "confidence": 0.0, "ear": 0.0, "mar": 0.0}

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

if not os.path.exists(PREDICTOR_PATH):
    print("[ERROR] Download shape_predictor_68_face_landmarks.dat")
    exit()

print("[INFO] Loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]


# [Keep all your existing functions: eye_aspect_ratio, mouth_aspect_ratio, etc.]
# I'm not repeating them here for brevity, but keep them as they are

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[1], mouth[7])
    B = dist.euclidean(mouth[2], mouth[6])
    C = dist.euclidean(mouth[3], mouth[5])
    D = dist.euclidean(mouth[0], mouth[4])
    return (A + B + C) / (3.0 * D)


def remove_outliers(values, factor=2.5):
    if len(values) < 3:
        return values
    mean = np.mean(values)
    std = np.std(values)
    if std == 0:
        return values
    z_scores = [(x - mean) / std for x in values]
    filtered = [x for x, z in zip(values, z_scores) if abs(z) < factor]
    return filtered if len(filtered) > 0 else values


def apply_smoothing(value, history):
    history.append(value)
    return np.mean(history) if len(history) > 0 else value


def interpolate_value(current, last, weight=0.7):
    return weight * current + (1 - weight) * last


def calculate_confidence(ear, mar, calibrated_ear, calibrated_mar_base, ear_std, mar_std, is_moving=False):
    confidence = 0.0
    movement_factor = 0.8 if is_moving else 1.0

    if ear < calibrated_ear and ear_std > 0:
        z_score = abs(calibrated_ear - ear) / ear_std
        eye_conf = min(1.0, z_score / 3.0)
        confidence += eye_conf * 0.5 * movement_factor

    if mar_std > 0:
        mar_deviation = abs(mar - calibrated_mar_base) / mar_std
        if mar_deviation > 2.0:
            mouth_conf = min(1.0, mar_deviation / 4.0)
            confidence += mouth_conf * 0.3 * movement_factor

    if len(ear_history) >= 3:
        stability = 1.0 - min(1.0, np.std(list(ear_history)[-3:]) * 10)
        confidence += stability * 0.2

    return min(1.0, confidence)


def detect_yawn_robust(current_mar, calibrated_mar_base, frame_time):
    """
    Robust yawn detection using state machine with temporal validation.
    Returns: (is_yawning, yawn_confidence, yawn_phase)
    """
    global yawn_state, yawn_start_time, yawn_peak_mar, yawn_frame_count
    global yawn_cooldown_counter, previous_mar, yawn_detected_count, mar_rate_history

    # Decrease cooldown counter
    if yawn_cooldown_counter > 0:
        yawn_cooldown_counter -= 1
        return False, 0.0, "cooldown"

    # Calculate MAR change rate
    mar_rate = current_mar - previous_mar
    mar_rate_history.append(mar_rate)
    previous_mar = current_mar

    # Define dynamic threshold based on calibrated baseline
    yawn_threshold = calibrated_mar_base * YAWN_MAR_INCREASE_THRESH

    is_yawning = False
    yawn_confidence = 0.0
    yawn_phase = yawn_state.lower()

    # State machine for yawn detection
    if yawn_state == "IDLE":
        # Check for mouth opening with sufficient rate
        if current_mar > yawn_threshold and mar_rate > YAWN_OPENING_RATE_THRESH:
            yawn_state = "OPENING"
            yawn_start_time = frame_time
            yawn_peak_mar = current_mar
            yawn_frame_count = 1

    elif yawn_state == "OPENING":
        yawn_frame_count += 1

        # Track peak MAR
        if current_mar > yawn_peak_mar:
            yawn_peak_mar = current_mar

        # Check if mouth is still opening or at peak
        if current_mar > yawn_threshold:
            # Check if rate is slowing down (approaching peak)
            if abs(mar_rate) < YAWN_OPENING_RATE_THRESH * 0.5:
                yawn_state = "PEAK"
        else:
            # Mouth closed too quickly - false positive
            yawn_state = "IDLE"
            yawn_start_time = None

    elif yawn_state == "PEAK":
        yawn_frame_count += 1

        # Update peak if still increasing
        if current_mar > yawn_peak_mar:
            yawn_peak_mar = current_mar

        # Check for closing motion (negative rate)
        if mar_rate < -YAWN_OPENING_RATE_THRESH * 0.3:
            yawn_state = "CLOSING"

        # Timeout if staying at peak too long
        if yawn_start_time and (frame_time - yawn_start_time) > YAWN_MAX_DURATION:
            yawn_state = "IDLE"
            yawn_start_time = None

    elif yawn_state == "CLOSING":
        yawn_frame_count += 1

        # Check if mouth has returned close to baseline
        if current_mar < calibrated_mar_base * 1.2:
            # Validate yawn duration
            if yawn_start_time:
                yawn_duration = frame_time - yawn_start_time

                if YAWN_MIN_DURATION <= yawn_duration <= YAWN_MAX_DURATION:
                    # Valid yawn detected!
                    is_yawning = True
                    yawn_detected_count += 1

                    # Calculate confidence based on duration and peak MAR
                    duration_score = min(1.0, yawn_duration / 3.0)
                    peak_score = min(1.0, (yawn_peak_mar / calibrated_mar_base - 1.0) / 0.5)
                    yawn_confidence = (duration_score * 0.6 + peak_score * 0.4)

                    # Set cooldown to prevent double-counting
                    yawn_cooldown_counter = YAWN_COOLDOWN_FRAMES

            # Reset state
            yawn_state = "IDLE"
            yawn_start_time = None
            yawn_frame_count = 0

        # Timeout check
        elif yawn_start_time and (frame_time - yawn_start_time) > YAWN_MAX_DURATION:
            yawn_state = "IDLE"
            yawn_start_time = None

    return is_yawning, yawn_confidence, yawn_phase


def enhanced_calibration(vs, detector, predictor):
    global eye_calibrated, mouth_calibrated, cal_eye_ratios, cal_mouth_ratios
    global calibrated_eye_threshold, calibrated_mouth_baseline, calibrated_mouth_threshold
    global eye_std_dev, mouth_std_dev

    print("\n[INFO] Starting calibration...")
    cal_eye_ratios = []
    cal_mouth_ratios = []
    stable_frames = 0

    while stable_frames < CALIBRATION_FRAMES:
        ret, frame = vs.read()
        if not ret:
            continue

        frame = cv2.resize(frame, (600, 400))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        progress = int((stable_frames / CALIBRATION_FRAMES) * 100)
        cv2.putText(frame, f"CALIBRATING... {progress}%", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if len(rects) > 0:
            rect = rects[0]
            face_width = rect.width()

            if face_width >= MIN_FACE_SIZE:
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                mouth = shape[mStart:mEnd]

                ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
                mar = mouth_aspect_ratio(mouth[12:20])

                cal_eye_ratios.append(ear)
                cal_mouth_ratios.append(mar)
                stable_frames += 1

                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # Update frame queue for web streaming during calibration
        try:
            while not frame_queue.empty():
                frame_queue.get_nowait()
            frame_queue.put(frame)
        except:
            pass

    # Process calibration
    filtered_ears = remove_outliers(cal_eye_ratios)
    filtered_mars = remove_outliers(cal_mouth_ratios)

    ear_percentile_25 = np.percentile(filtered_ears, 25)
    calibrated_eye_threshold = ear_percentile_25 * 0.75
    eye_std_dev = np.std(filtered_ears)

    calibrated_mouth_baseline = np.median(filtered_mars)
    calibrated_mouth_threshold = calibrated_mouth_baseline * 1.8
    mouth_std_dev = np.std(filtered_mars)

    eye_calibrated = True
    mouth_calibrated = True

    print(f"[INFO] Calibration complete!")
    ear_history.clear()
    mar_history.clear()


# Flask Web Server
app = Flask(__name__)


@app.route('/')
def index():
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>Drowsiness Detection System</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #1a1a1a;
            color: white;
        }
        .container {
            max-width: 1000px;
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
            background: #000;
        }
        .video-container img {
            max-width: 100%;
            height: auto;
        }
        .stats {
            display: flex;
            justify-content: space-around;
            margin: 20px 0;
        }
        .stat-box {
            background: #2a2a2a;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            flex: 1;
            margin: 0 10px;
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
        button:hover {
            background: #00dd00;
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
                        data.status.includes('Warning') || data.status.includes('YAWN') ? 'status-warning' : 'status-active';
                    document.getElementById('confidence').textContent = data.confidence.toFixed(2);
                    document.getElementById('ear').textContent = data.ear.toFixed(2);
                    document.getElementById('mar').textContent = data.mar.toFixed(2);
                    document.getElementById('yawn_count').textContent = data.yawn_count || 0;
                    document.getElementById('yawn_phase').textContent = data.yawn_phase || 'idle';
                });
        }

        function recalibrate() {
            fetch('/recalibrate', {method: 'POST'})
                .then(() => alert('Recalibration started'));
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
                <h3>EAR</h3>
                <h2 id="ear">0.00</h2>
            </div>
            <div class="stat-box">
                <h3>MAR</h3>
                <h2 id="mar">0.00</h2>
            </div>
            <div class="stat-box">
                <h3>Yawn Count</h3>
                <h2 id="yawn_count">0</h2>
            </div>
            <div class="stat-box">
                <h3>Yawn Phase</h3>
                <h2 id="yawn_phase">idle</h2>
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
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except:
                continue

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status')
def status():
    return jsonify(current_status)


@app.route('/recalibrate', methods=['POST'])
def recalibrate():
    global need_recalibration
    need_recalibration = True
    return jsonify({'status': 'recalibration_started'})


need_recalibration = False


def main_detection_loop():
    global need_recalibration, current_status, last_ear, last_mar

    print("[INFO] Starting movement-tolerant detection...")
    vs = cv2.VideoCapture(0)
    time.sleep(2.0)

    # Initial calibration
    enhanced_calibration(vs, detector, predictor)

    frame_count = 0
    no_face_frames = 0

    while True:
        if need_recalibration:
            enhanced_calibration(vs, detector, predictor)
            need_recalibration = False

        ret, frame = vs.read()
        if not ret:
            continue

        frame = cv2.resize(frame, (600, 400))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        status_text = "Active"
        color = (0, 255, 0)
        is_moving = False

        if len(rects) == 0:
            no_face_frames += 1
            if no_face_frames < 10:
                ear = last_ear
                mar = last_mar
                status_text = "Tracking..." if no_face_frames < 5 else "Searching..."
                color = (255, 255, 0)
                confidence_score = confidence_score * 0.98 if 'confidence_score' in locals() else 0
            else:
                status_text = "Face Lost"
                color = (128, 128, 128)
                confidence_score = 0
        else:
            no_face_frames = 0
            rect = max(rects, key=lambda r: r.width() * r.height())
            face_width = rect.width()

            if face_width >= MIN_FACE_SIZE:
                if last_face_width:
                    size_change = abs(face_width - last_face_width) / last_face_width
                    if size_change > MAX_FACE_MOVEMENT:
                        is_moving = True

                last_face_width = face_width

                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                mouth = shape[mStart:mEnd]

                ear_raw = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
                mar_raw = mouth_aspect_ratio(mouth[12:20])

                if is_moving:
                    ear_interp = interpolate_value(ear_raw, last_ear, INTERPOLATION_WEIGHT)
                    mar_interp = interpolate_value(mar_raw, last_mar, INTERPOLATION_WEIGHT)
                else:
                    ear_interp = ear_raw
                    mar_interp = mar_raw

                last_ear = ear_interp
                last_mar = mar_interp

                ear = apply_smoothing(ear_interp, ear_history)
                mar = apply_smoothing(mar_interp, mar_history)

                confidence_score = calculate_confidence(
                    ear, mar, calibrated_eye_threshold, calibrated_mouth_baseline,
                    eye_std_dev, mouth_std_dev, is_moving
                )

                # Robust yawn detection
                current_time = time.time()
                is_yawning, yawn_confidence, yawn_phase = detect_yawn_robust(
                    mar, calibrated_mouth_baseline, current_time
                )

                # Draw contours
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                mouthHull = cv2.convexHull(mouth)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                # Color mouth based on yawn phase
                if yawn_phase == "opening":
                    cv2.drawContours(frame, [mouthHull], -1, (0, 255, 255), 2)  # Yellow
                elif yawn_phase == "peak":
                    cv2.drawContours(frame, [mouthHull], -1, (0, 165, 255), 2)  # Orange
                elif yawn_phase == "closing":
                    cv2.drawContours(frame, [mouthHull], -1, (255, 0, 255), 2)  # Magenta
                else:
                    cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)  # Green

                # Detection logic
                eyes_closed = ear < calibrated_eye_threshold
                yawn_detected = is_yawning  # Use robust yawn detection
                mouth_not_moving = abs(mar - calibrated_mouth_baseline) < MOUTH_CHANGE_THRESH

                min_conf = MIN_CONFIDENCE_THRESHOLD * 0.8 if is_moving else MIN_CONFIDENCE_THRESHOLD

                global EYE_CLOSED_COUNTER, DROWSY_ALERT

                if confidence_score > min_conf:
                    if eyes_closed and (yawn_detected or mouth_not_moving):
                        EYE_CLOSED_COUNTER += 1
                        if EYE_CLOSED_COUNTER >= EYE_AR_CONSEC_FRAMES:
                            DROWSY_ALERT = True
                    else:
                        EYE_CLOSED_COUNTER = max(0, EYE_CLOSED_COUNTER - 2)
                        if EYE_CLOSED_COUNTER < EYE_AR_CONSEC_FRAMES // 2:
                            DROWSY_ALERT = False
                else:
                    EYE_CLOSED_COUNTER = max(0, EYE_CLOSED_COUNTER - 1)

                if DROWSY_ALERT:
                    status_text = "DROWSINESS ALERT!"
                    color = (0, 0, 255)
                elif is_yawning:
                    status_text = "YAWN DETECTED!"
                    color = (0, 165, 255)
                elif EYE_CLOSED_COUNTER > EYE_AR_CONSEC_FRAMES // 2:
                    status_text = "Warning"
                    color = (0, 165, 255)
                elif yawn_phase != "idle" and yawn_phase != "cooldown":
                    status_text = f"Active (Yawn {yawn_phase})"
                    color = (0, 255, 255)
                elif is_moving:
                    status_text = "Active (Moving)"
                    color = (0, 255, 0)

                # Update current status
                current_status = {
                    "status": status_text,
                    "confidence": float(confidence_score),
                    "ear": float(ear),
                    "mar": float(mar),
                    "yawn_count": yawn_detected_count,
                    "yawn_phase": yawn_phase
                }

        # Add status text to frame
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Conf: {confidence_score:.2f}", (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Display yawn counter if any yawns detected
        if 'yawn_detected_count' in globals() and yawn_detected_count > 0:
            cv2.putText(frame, f"Yawns: {yawn_detected_count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Put frame in queue for web streaming
        try:
            while not frame_queue.empty():
                frame_queue.get_nowait()
            frame_queue.put(frame)
        except:
            pass


if __name__ == "__main__":
    # Start detection in separate thread
    detection_thread = threading.Thread(target=main_detection_loop)
    detection_thread.daemon = True
    detection_thread.start()

    # Start Flask web server
    print("\n[INFO] Starting web interface...")
    print("[INFO] Open browser at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)