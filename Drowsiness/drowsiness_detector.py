# Robust Drowsiness Detector with Enhanced Calibration and Noise Filtering
# Optimized for real-world bus environments with varying conditions
# This script detects drowsiness based on a combination of eye closure
# and mouth state (yawning or lack of expression change).
# Features adaptive calibration for different faces and noise filtering.

import cv2
import dlib
from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import os
from collections import deque
import time

# Define constants for eye and mouth detection
EYE_AR_THRESH = 0.25  # Initial eye aspect ratio threshold (will be calibrated)
EYE_AR_CONSEC_FRAMES = 60  # Increased from 48 for robustness
YAWN_MAR_THRESH = 0.6  # Initial mouth aspect ratio threshold for a yawn
MOUTH_CHANGE_THRESH = 0.05  # Threshold to detect "no change" in mouth

# Constants for enhanced calibration
CALIBRATION_FRAMES = 100  # Increased from 60 for better statistics
CALIBRATION_TEXT = "Calibrating... Please keep eyes open, mouth closed"

# Confidence and filtering constants
MIN_CONFIDENCE_THRESHOLD = 0.65  # Minimum confidence to trigger alert
SMOOTH_WINDOW_SIZE = 5  # Moving average window for smoothing
OUTLIER_FACTOR = 2.0  # Standard deviations for outlier rejection
MIN_FACE_SIZE = 80  # Minimum face width in pixels

# Initialize frame counters and state variables
EYE_CLOSED_COUNTER = 0
DROWSY_ALERT = False
confidence_score = 0.0

# Variables for calibration
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

# Path to the facial landmark predictor model
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

# Check if the predictor file exists
if not os.path.exists(PREDICTOR_PATH):
    print("[ERROR] Facial landmark predictor file not found.")
    print(f"Please download '{PREDICTOR_PATH}' from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 and extract it.")
    exit()

# Load dlib's face detector and then create the facial landmark predictor
print("[INFO] Loading facial landmark predictor...")
try:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
except Exception as e:
    print(f"[ERROR] Failed to load dlib predictor: {e}")
    exit()

# Get the indexes of the facial landmarks for the left and right eyes and mouth
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]


def eye_aspect_ratio(eye):
    """Compute the eye aspect ratio"""
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def mouth_aspect_ratio(mouth):
    """Compute the mouth aspect ratio"""
    A = dist.euclidean(mouth[1], mouth[7])
    B = dist.euclidean(mouth[2], mouth[6])
    C = dist.euclidean(mouth[3], mouth[5])
    D = dist.euclidean(mouth[0], mouth[4])
    mar = (A + B + C) / (3.0 * D)
    return mar


def remove_outliers(values, factor=2.0):
    """Remove outliers using z-score method"""
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
    """Apply moving average smoothing"""
    history.append(value)
    return np.mean(history) if len(history) > 0 else value


def calculate_confidence(ear, mar, calibrated_ear, calibrated_mar_base, ear_std, mar_std):
    """Calculate confidence score for drowsiness detection"""
    confidence = 0.0
    
    # Eye closure confidence
    if ear < calibrated_ear and ear_std > 0:
        z_score = abs(calibrated_ear - ear) / ear_std
        eye_conf = min(1.0, z_score / 3.0)
        confidence += eye_conf * 0.5
    
    # Mouth state confidence
    if mar_std > 0:
        mar_deviation = abs(mar - calibrated_mar_base) / mar_std
        if mar_deviation > 2.0:  # Significant change (yawn or no movement)
            mouth_conf = min(1.0, mar_deviation / 4.0)
            confidence += mouth_conf * 0.3
    
    # Add base confidence if measurements are stable
    if len(ear_history) == SMOOTH_WINDOW_SIZE:
        stability = 1.0 - min(1.0, np.std(ear_history) * 10)
        confidence += stability * 0.2
    
    return min(1.0, confidence)


def enhanced_calibration(vs, detector, predictor):
    """Enhanced calibration with outlier rejection and statistics"""
    global eye_calibrated, mouth_calibrated, cal_eye_ratios, cal_mouth_ratios
    global calibrated_eye_threshold, calibrated_mouth_baseline, calibrated_mouth_threshold
    global eye_std_dev, mouth_std_dev
    
    print("\n[INFO] Starting enhanced calibration...")
    print("[INFO] Please look naturally at the camera with eyes open and mouth closed")
    
    cal_eye_ratios = []
    cal_mouth_ratios = []
    stable_frames = 0
    total_frames = 0
    face_sizes = []
    
    while stable_frames < CALIBRATION_FRAMES and total_frames < CALIBRATION_FRAMES * 2:
        ret, frame = vs.read()
        if not ret:
            continue
        
        total_frames += 1
        frame = cv2.resize(frame, (600, 400))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        
        progress = int((stable_frames / CALIBRATION_FRAMES) * 100)
        cv2.putText(frame, f"CALIBRATING... {progress}%", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, "Keep eyes open, mouth closed", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        if len(rects) > 0:
            rect = rects[0]
            face_width = rect.width()
            
            # Check minimum face size
            if face_width < MIN_FACE_SIZE:
                cv2.putText(frame, "Please move closer", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.imshow("Drowsiness Detection", frame)
                cv2.waitKey(1)
                continue
            
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            mouth = shape[mStart:mEnd]
            
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            mar = mouth_aspect_ratio(mouth[12:20])  # Inner mouth points
            
            # Check for stability before collecting
            if len(ear_history) > 0:
                recent_avg = np.mean(list(ear_history)[-3:]) if len(ear_history) >= 3 else ear_history[-1]
                if abs(ear - recent_avg) / recent_avg < 0.15:  # Within 15% of recent average
                    cal_eye_ratios.append(ear)
                    cal_mouth_ratios.append(mar)
                    face_sizes.append(face_width)
                    stable_frames += 1
            else:
                ear_history.append(ear)
            
            # Draw contours for visual feedback
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            mouthHull = cv2.convexHull(mouth)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
        
        cv2.imshow("Drowsiness Detection", frame)
        cv2.waitKey(1)
    
    if len(cal_eye_ratios) >= CALIBRATION_FRAMES * 0.8:
        # Remove outliers
        filtered_ears = remove_outliers(cal_eye_ratios, OUTLIER_FACTOR)
        filtered_mars = remove_outliers(cal_mouth_ratios, OUTLIER_FACTOR)
        
        # Calculate robust statistics using percentiles
        ear_percentile_25 = np.percentile(filtered_ears, 25)
        calibrated_eye_threshold = ear_percentile_25 * 0.75  # Conservative multiplier
        eye_std_dev = np.std(filtered_ears)
        
        calibrated_mouth_baseline = np.median(filtered_mars)
        calibrated_mouth_threshold = calibrated_mouth_baseline * 1.8  # Adaptive yawn threshold
        mouth_std_dev = np.std(filtered_mars)
        
        eye_calibrated = True
        mouth_calibrated = True
        
        print(f"[INFO] Calibration complete!")
        print(f"  EAR Threshold: {calibrated_eye_threshold:.3f} (std: {eye_std_dev:.3f})")
        print(f"  MAR Baseline: {calibrated_mouth_baseline:.3f}")
        print(f"  MAR Yawn Threshold: {calibrated_mouth_threshold:.3f} (std: {mouth_std_dev:.3f})")
    else:
        print("[WARNING] Calibration failed: Insufficient stable samples")
        print("[INFO] Using conservative default values")
        calibrated_eye_threshold = 0.20
        calibrated_mouth_baseline = 0.30
        calibrated_mouth_threshold = 0.65
        eye_std_dev = 0.05
        mouth_std_dev = 0.10
        eye_calibrated = True
        mouth_calibrated = True
    
    # Clear histories after calibration
    ear_history.clear()
    mar_history.clear()
    cal_eye_ratios = []
    cal_mouth_ratios = []


# Start video stream
print("[INFO] Starting video stream...")
vs = cv2.VideoCapture(0)

if not vs.isOpened():
    print("[ERROR] Could not open video stream. Please check your webcam.")
    exit()

# Wait for camera warm-up
cv2.waitKey(2000)

# Perform initial calibration
enhanced_calibration(vs, detector, predictor)

# Main detection loop
frame_count = 0
no_face_frames = 0

while True:
    ret, frame = vs.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (600, 400))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    rects = detector(gray, 0)
    
    # Handle no face detected
    if len(rects) == 0:
        no_face_frames += 1
        if no_face_frames > 30:
            # Reset counters if face lost for too long
            EYE_CLOSED_COUNTER = 0
            DROWSY_ALERT = False
            confidence_score *= 0.95  # Decay confidence
    else:
        no_face_frames = 0
        
        # Use the largest face (likely the driver)
        rect = max(rects, key=lambda r: r.width() * r.height())
        face_width = rect.width()
        
        # Skip if face too small
        if face_width < MIN_FACE_SIZE:
            cv2.putText(frame, "Face too far", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            cv2.imshow("Drowsiness Detection", frame)
            cv2.waitKey(1)
            continue
        
        # Check for sudden face size changes (movement/shake)
        if last_face_width and abs(face_width - last_face_width) / last_face_width > 0.3:
            # Ignore this frame due to sudden movement
            last_face_width = face_width
            continue
        last_face_width = face_width
        
        # Get facial landmarks
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # Extract eye and mouth coordinates
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]
        
        # Calculate raw EAR and MAR
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear_raw = (leftEAR + rightEAR) / 2.0
        mar_raw = mouth_aspect_ratio(mouth[12:20])
        
        # Apply smoothing
        ear = apply_smoothing(ear_raw, ear_history)
        mar = apply_smoothing(mar_raw, mar_history)
        
        # Calculate confidence score
        confidence_score = calculate_confidence(
            ear, mar, calibrated_eye_threshold, calibrated_mouth_baseline,
            eye_std_dev, mouth_std_dev
        )
        
        # Draw contours around eyes and mouth for visualization
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
        
        # Enhanced drowsiness detection logic with confidence
        eyes_closed = ear < calibrated_eye_threshold
        yawn_detected = mar > calibrated_mouth_threshold
        mouth_not_moving = abs(mar - calibrated_mouth_baseline) < MOUTH_CHANGE_THRESH
        
        # Only process if confidence is sufficient
        if confidence_score > MIN_CONFIDENCE_THRESHOLD:
            if eyes_closed and (yawn_detected or mouth_not_moving):
                EYE_CLOSED_COUNTER += 1
                if EYE_CLOSED_COUNTER >= EYE_AR_CONSEC_FRAMES:
                    DROWSY_ALERT = True
            else:
                # Gradual recovery
                EYE_CLOSED_COUNTER = max(0, EYE_CLOSED_COUNTER - 2)
                if EYE_CLOSED_COUNTER < EYE_AR_CONSEC_FRAMES // 2:
                    DROWSY_ALERT = False
        else:
            # Low confidence - slowly decay counter
            EYE_CLOSED_COUNTER = max(0, EYE_CLOSED_COUNTER - 1)
        
        # Display the status and information on the frame
        if DROWSY_ALERT:
            status_text = "DROWSINESS ALERT!"
            color = (0, 0, 255)  # Red
        elif EYE_CLOSED_COUNTER > EYE_AR_CONSEC_FRAMES // 2:
            status_text = "Warning"
            color = (0, 165, 255)  # Orange
        else:
            status_text = "Active"
            color = (0, 255, 0)  # Green
        
        cv2.putText(frame, status_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"EAR: {ear:.2f}", (450, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (450, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"Conf: {confidence_score:.2f}", (450, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Show the frame
    cv2.imshow("Drowsiness Detection", frame)
    key = cv2.waitKey(1) & 0xFF
    
    # 'q' to quit, 'r' to recalibrate
    if key == ord("q"):
        break
    elif key == ord("r"):
        print("\n[INFO] Manual recalibration requested")
        enhanced_calibration(vs, detector, predictor)

# Cleanup
cv2.destroyAllWindows()
vs.release()