#!/usr/bin/env python3
"""Quick diagnostic test for camera and MediaPipe face detection"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

print("=" * 60)
print("CAMERA & MEDIAPIPE DIAGNOSTIC TEST")
print("=" * 60)

# Test 1: Check if camera opens
print("\n[TEST 1] Opening camera 0...")
vs = cv2.VideoCapture(0)
if not vs.isOpened():
    print("❌ FAILED: Could not open camera 0")
    exit(1)
else:
    print("✓ SUCCESS: Camera 0 opened")

# Test 2: Read a frame
print("\n[TEST 2] Reading frame from camera...")
ret, frame = vs.read()
if not ret or frame is None:
    print("❌ FAILED: Could not read frame from camera")
    vs.release()
    exit(1)
else:
    print(f"✓ SUCCESS: Frame captured - Shape: {frame.shape}, Mean brightness: {frame.mean():.1f}")

# Test 3: Initialize MediaPipe
print("\n[TEST 3] Initializing MediaPipe Face Landmarker...")
model_path = 'face_landmarker_v2_with_blendshapes.task'
if not os.path.exists(model_path):
    print(f"❌ FAILED: Model file not found: {model_path}")
    vs.release()
    exit(1)

try:
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True
    )
    face_landmarker = vision.FaceLandmarker.create_from_options(options)
    print("✓ SUCCESS: MediaPipe initialized")
except Exception as e:
    print(f"❌ FAILED: MediaPipe initialization error: {e}")
    vs.release()
    exit(1)

# Test 4: Try face detection with and without preprocessing
print("\n[TEST 4] Testing face detection...")

for test_num in range(10):
    ret, frame = vs.read()
    if not ret:
        continue

    # Test 4a: Original frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = face_landmarker.detect(mp_image)

    original_detected = len(result.face_landmarks) > 0 if result.face_landmarks else False

    # Test 4b: Preprocessed frame (LAB color space enhancement)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    rgb_processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    mp_image_processed = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_processed)
    result_processed = face_landmarker.detect(mp_image_processed)

    processed_detected = len(result_processed.face_landmarks) > 0 if result_processed.face_landmarks else False

    status_orig = "✓" if original_detected else "✗"
    status_proc = "✓" if processed_detected else "✗"

    print(f"  Frame {test_num + 1}/10: Original={status_orig} Preprocessed={status_proc}")

    if original_detected or processed_detected:
        break

vs.release()

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
print("\nIf you see ✗ for all frames, the issue is:")
print("1. No face in front of camera")
print("2. Camera is too dark/bright")
print("3. MediaPipe confidence thresholds too high")
print("\nIf you see ✓ here but calibration fails in main script,")
print("the issue is in the calibration loop logic.")
