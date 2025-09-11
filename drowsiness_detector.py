# Drowsiness Detector with Calibration and Specific Logic
# This script detects drowsiness based on a combination of eye closure
# and mouth state (yawning or lack of expression change).
# It requires initial calibration from the user.

# To run this script, you will need to install the following libraries:
# pip install opencv-python dlib imutils scipy

# NOTE: You must download the pre-trained facial landmark predictor model
# from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
# and extract it. Place the `shape_predictor_68_face_landmarks.dat` file
# in the same directory as this script.

import cv2
import dlib
from scipy.spatial import distance as dist
from imutils import face_utils
import os # Added for file path checking

# Define constants for eye and mouth detection
EYE_AR_THRESH = 0.25 # Initial eye aspect ratio threshold
EYE_AR_CONSEC_FRAMES = 48 # Number of consecutive frames eyes must be closed
YAWN_MAR_THRESH = 0.6 # Mouth aspect ratio threshold for a yawn
MOUTH_CHANGE_THRESH = 0.05 # Threshold to detect "no change" in mouth

# Constants for calibration
CALIBRATION_FRAMES = 60 # Number of frames to average for calibration
CALIBRATION_TEXT = "Calibrating..."

# Initialize frame counters and state variables
EYE_CLOSED_COUNTER = 0
DROWSY_ALERT = False

# Variables for calibration
eye_calibrated = False
mouth_calibrated = False
cal_eye_ratios = []
cal_mouth_ratios = []
calibrated_eye_threshold = 0.0
calibrated_mouth_threshold = 0.0

# Path to the facial landmark predictor model
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

# Check if the predictor file exists
if not os.path.exists(PREDICTOR_PATH):
    print("[ERROR] Facial landmark predictor file not found.")
    print(f"Please download '{PREDICTOR_PATH}' from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 and extract it.")
    exit()

# Load dlib's face detector and then create the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
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
    # compute the euclidean distances between the two sets of vertical eye landmarks
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal eye landmark
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    return ear

def mouth_aspect_ratio(mouth):
    # compute the euclidean distances between the two sets of vertical mouth landmarks
    A = dist.euclidean(mouth[1], mouth[7])
    B = dist.euclidean(mouth[2], mouth[6])
    C = dist.euclidean(mouth[3], mouth[5])

    # compute the euclidean distance between the horizontal mouth landmark
    D = dist.euclidean(mouth[0], mouth[4])

    # compute the mouth aspect ratio
    mar = (A + B + C) / (3.0 * D)

    return mar


# Start video stream
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(0)

if not vs.isOpened():
    print("[ERROR] Could not open video stream. Please check your webcam.")
    exit()

# Wait for a brief moment for the camera to warm up
cv2.waitKey(2000)

while True:
    # grab the frame from the threaded video stream
    ret, frame = vs.read()
    if not ret:
        break

    # resize the frame and convert it to grayscale
    frame = cv2.resize(frame, (600, 400))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    # loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # extract the eye and mouth coordinates
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]

        # calculate the EAR and MAR for both eyes and mouth
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        # Corrected line: mouth[12:20] corresponds to original facial landmarks 60-67
        mar = mouth_aspect_ratio(mouth[12:20])

        # Draw contours around eyes and mouth for visualization
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

        # Check for calibration state
        if not eye_calibrated:
            cal_eye_ratios.append(ear)
            if len(cal_eye_ratios) >= CALIBRATION_FRAMES:
                calibrated_eye_threshold = sum(cal_eye_ratios) / len(cal_eye_ratios) * 0.85 # A slight buffer
                eye_calibrated = True
                print(f"[INFO] Eye calibration complete. Threshold: {calibrated_eye_threshold:.2f}")
                cal_eye_ratios = []
        
        if not mouth_calibrated:
            cal_mouth_ratios.append(mar)
            if len(cal_mouth_ratios) >= CALIBRATION_FRAMES:
                calibrated_mouth_threshold = sum(cal_mouth_ratios) / len(cal_mouth_ratios)
                mouth_calibrated = True
                print(f"[INFO] Mouth calibration complete. Baseline MAR: {calibrated_mouth_threshold:.2f}")
                cal_mouth_ratios = []

        # Wait until calibration is complete before proceeding to detection
        if eye_calibrated and mouth_calibrated:
            # Check for drowsy state based on user's logic
            # Condition 1: Eyes are closed
            eyes_closed = ear < calibrated_eye_threshold
            
            # Condition 2: Mouth state is either a yawn or no change
            yawn_detected = mar > YAWN_MAR_THRESH
            
            mouth_not_moving = abs(mar - calibrated_mouth_threshold) < MOUTH_CHANGE_THRESH
            
            if eyes_closed and (yawn_detected or mouth_not_moving):
                EYE_CLOSED_COUNTER += 1
                if EYE_CLOSED_COUNTER >= EYE_AR_CONSEC_FRAMES:
                    DROWSY_ALERT = True
            else:
                EYE_CLOSED_COUNTER = 0
                DROWSY_ALERT = False

        # Display the status and information on the frame
        if not eye_calibrated or not mouth_calibrated:
            status_text = "Calibrating..."
            color = (0, 255, 255) # Yellow
        elif DROWSY_ALERT:
            status_text = "DROWSINESS ALERT!"
            color = (0, 0, 255) # Red
        else:
            status_text = "Active"
            color = (0, 255, 0) # Green
            
        cv2.putText(frame, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"EAR: {ear:.2f}", (450, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (450, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow("Drowsiness Detection", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key was pressed, break from the loop
    if key == ord("q"):
        break

# cleanup
cv2.destroyAllWindows()
vs.release()
