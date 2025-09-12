import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
import time
import os

# --- SETTINGS (Adjust these to tune sensitivity) ---

# Path to the dlib facial landmark predictor model
DLIB_LANDMARK_MODEL = "shape_predictor_68_face_landmarks.dat"

# Use 0 for a local webcam or provide the path/URL to a video file
CAMERA_SOURCE = 0

# Number of frames to use for the initial calibration
CALIBRATION_FRAMES = 60

# --- DROWSINESS THRESHOLDS ---
EAR_PERCENTAGE_THRESHOLD = 0.85
# 1. Time-based alert for long eye closures
EYE_CLOSURE_SECONDS_THRESHOLD = 2.0
# 2. Frame-based alert for shorter, consecutive eye closures
DROWSY_CONSEC_FRAMES = 25

# --- YAWN DETECTION THRESHOLDS (NOW WITH DURATION) ---
# Mouth Aspect Ratio threshold to be considered "open wide"
YAWN_MAR_THRESHOLD = 0.9
# How many seconds the mouth must be "open wide" to be confirmed as a yawn
YAWN_DURATION_SECONDS = 2.5  # <<< NEW SETTING

# --- SMILE DETECTION THRESHOLD ---
# Ratio of mouth width to eye distance. Higher means a wider smile.
SMILE_THRESHOLD = 0.70


# --- END OF SETTINGS ---


def eye_aspect_ratio(eye):
    """Computes the Eye Aspect Ratio (EAR) for a single eye."""
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def mouth_aspect_ratio(mouth):
    """Computes the Mouth Aspect Ratio (MAR)."""
    A = dist.euclidean(mouth[2], mouth[10])  # Vertical
    B = dist.euclidean(mouth[4], mouth[8])  # Vertical
    C = dist.euclidean(mouth[0], mouth[6])  # Horizontal
    mar = (A + B) / (2.0 * C)
    return mar


# --- STATE VARIABLES ---
is_calibrated = False
calibration_ear_values = []
calibrated_ear_threshold = 0.0
eye_closure_start_time = None  # For time-based check
drowsy_counter = 0  # For frame-based check
yawn_start_time = None  # For yawn timer check
# --- END STATE VARIABLES ---

# Check if the model file exists
if not os.path.exists(DLIB_LANDMARK_MODEL):
    print(f"[ERROR] Dlib model not found at '{DLIB_LANDMARK_MODEL}'")
    exit()

print("[INFO] Loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(DLIB_LANDMARK_MODEL)

# Get landmark indexes
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

print("[INFO] Starting video stream...")
vs = cv2.VideoCapture(CAMERA_SOURCE)
time.sleep(1.0)

while True:
    ret, frame = vs.read()
    if not ret:
        break

    frame = cv2.resize(frame, (800, 600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    status_text = "Active"
    alert_triggered = False

    if len(rects) > 0:
        shape = predictor(gray, rects[0])
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]

        ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
        mar = mouth_aspect_ratio(mouth)

        # Smile Detection
        smile_width = dist.euclidean(shape[48], shape[54])
        eye_distance = dist.euclidean(shape[36], shape[45])
        smile_ratio = smile_width / eye_distance
        is_smiling = smile_ratio > SMILE_THRESHOLD

        # Draw contours for visualization
        cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (0, 255, 0), 1)

        if not is_calibrated:
            status_text = "Calibrating..."
            calibration_ear_values.append(ear)
            if len(calibration_ear_values) >= CALIBRATION_FRAMES:
                median_ear = np.median(calibration_ear_values)
                calibrated_ear_threshold = median_ear * EAR_PERCENTAGE_THRESHOLD
                is_calibrated = True
                print(f"[INFO] Calibration complete. EAR Threshold set to: {calibrated_ear_threshold:.2f}")
        else:
            is_eyes_closed = ear < calibrated_ear_threshold
            is_mouth_open_wide = mar > YAWN_MAR_THRESHOLD

            # --- COMBINED DROWSINESS LOGIC ---

            # 1. Check for a sustained yawn
            if is_mouth_open_wide and not is_smiling:
                if yawn_start_time is None:
                    yawn_start_time = time.time()
                else:
                    elapsed_time = time.time() - yawn_start_time
                    if elapsed_time >= YAWN_DURATION_SECONDS:
                        alert_triggered = True
            else:
                yawn_start_time = None

            # 2. Check for eye closure if not yawning
            if not alert_triggered and is_eyes_closed and not is_smiling:
                # Time-based check
                if eye_closure_start_time is None:
                    eye_closure_start_time = time.time()
                else:
                    elapsed_time = time.time() - eye_closure_start_time
                    if elapsed_time >= EYE_CLOSURE_SECONDS_THRESHOLD:
                        alert_triggered = True

                # Frame-based check
                drowsy_counter += 1
                if drowsy_counter >= DROWSY_CONSEC_FRAMES:
                    alert_triggered = True
            else:
                # Reset eye counters if eyes are open or smiling
                eye_closure_start_time = None
                drowsy_counter = 0

            if alert_triggered:
                status_text = "DROWSINESS ALERT!"
            elif is_smiling:
                status_text = "Active (Smiling)"

    else:
        status_text = "No Face Detected"
        # Reset all counters if face is lost
        eye_closure_start_time = None
        drowsy_counter = 0
        yawn_start_time = None

    # Draw the status text
    color = (0, 0, 255) if alert_triggered else (0, 255, 0)
    cv2.putText(frame, f"Status: {status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Drowsiness Detector", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.release()