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

        # --- THRESHOLDS ---
        self.EAR_PERCENTAGE_THRESHOLD = 0.85
        self.EYE_CLOSURE_SECONDS_THRESHOLD = 2.0
        self.DROWSY_CONSEC_FRAMES = 25
        self.YAWN_MAR_THRESHOLD = 0.7
        self.YAWN_DURATION_SECONDS = 1.5
        self.SMILE_THRESHOLD = 0.8

        # --- BLINK & FATIGUE DETECTION ---
        self.BLINK_CONSEC_FRAMES = 3
        self.FATIGUE_BLINK_RATE_SECONDS = 30
        self.FATIGUE_LOW_BLINK_THRESHOLD = 8
        self.FATIGUE_HIGH_BLINK_THRESHOLD = 40
        self.FATIGUE_EVENT_WINDOW_SECONDS = 120  # <<< NEW
        self.FATIGUE_EVENT_THRESHOLD = 3  # <<< NEW

        # --- HEAD NOD DETECTION ---
        self.HEAD_NOD_FRAME_WINDOW = 20
        self.HEAD_NOD_THRESHOLD_PIXELS = 18

        # --- RECALIBRATION SETTINGS ---
        self.RECALIBRATION_FRAMES = 20
        self.RECALIBRATION_INTERVAL_SECONDS = 60
        self.FACE_TRACKING_TOLERANCE_PIXELS = 75
        self.RECALIBRATION_CANDIDATE_SECONDS = 3.0

        # --- STATE MANAGEMENT (NEW) ---
        self.STATE = {"ACTIVE": 0, "FATIGUED": 1, "DROWSY": 2}
        self.current_state = self.STATE["ACTIVE"]

        # --- ALL COUNTERS AND STATE VARIABLES ---
        self.calibrated_ear_threshold = None
        self.is_calibrating = False
        self.calibration_ear_values = []
        self.last_recalibration_time = None
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
        self.head_positions = []
        self.is_yawning = False
        self.is_nodding = False
        self.fatigue_events = []  # <<< NEW

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

    def recalibrate(self, vs):
        print(f"\n[{time.ctime()}] Starting recalibration...")
        self.is_calibrating = True
        self.calibration_ear_values = []
        for i in range(self.RECALIBRATION_FRAMES):
            ret, frame = vs.read()
            if not ret: continue
            frame = cv2.resize(frame, (800, 600))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = self.detector(gray, 0)
            cv2.putText(frame, "RECALIBRATING...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            if len(rects) > 0:
                shape = self.predictor(gray, rects[0])
                shape = face_utils.shape_to_np(shape)
                leftEye = shape[self.lStart:self.lEnd]
                rightEye = shape[self.rStart:self.rEnd]
                ear = (self.eye_aspect_ratio(leftEye) + self.eye_aspect_ratio(rightEye)) / 2.0
                self.calibration_ear_values.append(ear)
            cv2.imshow("Drowsiness Detector", frame)
            cv2.waitKey(1)
        if self.calibration_ear_values:
            median_ear = np.median(self.calibration_ear_values)
            self.calibrated_ear_threshold = median_ear * self.EAR_PERCENTAGE_THRESHOLD
            print(f"[{time.ctime()}] Recalibration complete. New EAR Threshold: {self.calibrated_ear_threshold:.2f}\n")
        else:
            print("[WARNING] Recalibration failed: No face detected.")
        self.is_calibrating = False
        self.last_recalibration_time = time.time()
        self.reset_counters()

    def reset_counters(self):
        self.eye_closure_start_time = None
        self.drowsy_counter = 0
        self.yawn_start_time = None
        self.yawn_count = 0
        self.nod_count = 0
        self.blink_counter = 0
        self.fatigue_events = []

    def run(self):
        print("[INFO] Starting video stream...")
        vs = cv2.VideoCapture(self.CAMERA_SOURCE)
        time.sleep(1.0)
        self.recalibrate(vs)

        while True:
            if not self.is_calibrating and time.time() - self.last_recalibration_time > self.RECALIBRATION_INTERVAL_SECONDS:
                self.recalibrate(vs)

            ret, frame = vs.read()
            if not ret: break

            frame = cv2.resize(frame, (800, 600))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = self.detector(gray, 0)

            status_text = "No Face Detected"  # <<< MODIFIED
            color = (0, 255, 0)  # <<< MODIFIED

            if len(rects) > 0:
                largest_rect = max(rects, key=lambda rect: rect.width() * rect.height())

                (x, y, w, h) = (largest_rect.left(), largest_rect.top(), largest_rect.width(), largest_rect.height())
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

                if self.is_calibrating:
                    status_text = "Calibrating..."
                    color = (255, 255, 0)
                else:
                    shape = self.predictor(gray, largest_rect)
                    shape = face_utils.shape_to_np(shape)

                    leftEye, rightEye, mouth, nose = shape[self.lStart:self.lEnd], shape[self.rStart:self.rEnd], shape[
                        self.mStart:self.mEnd], shape[self.nStart:self.nEnd]
                    ear = (self.eye_aspect_ratio(leftEye) + self.eye_aspect_ratio(rightEye)) / 2.0
                    mar = self.mouth_aspect_ratio(mouth)
                    #print(f"Current MAR: {mar:.2f}")  # <-- ADD THIS LINE

                    is_eyes_closed = ear < self.calibrated_ear_threshold
                    is_mouth_open_wide = mar > self.YAWN_MAR_THRESHOLD

                    # --- START OF MODIFIED LOGIC ---

                    # --- Fatigue Analysis ---
                    if is_eyes_closed:
                        self.blink_frame_counter += 1
                    else:
                        if self.blink_frame_counter >= self.BLINK_CONSEC_FRAMES: self.blink_counter += 1
                        self.blink_frame_counter = 0

                    fatigue_detected = False
                    if time.time() - self.blink_analysis_start_time > self.FATIGUE_BLINK_RATE_SECONDS:
                        self.display_blink_rate = (self.blink_counter / self.FATIGUE_BLINK_RATE_SECONDS) * 60
                        if self.display_blink_rate < self.FATIGUE_LOW_BLINK_THRESHOLD or self.display_blink_rate > self.FATIGUE_HIGH_BLINK_THRESHOLD:
                            fatigue_detected = True
                        self.blink_counter = 0
                        self.blink_analysis_start_time = time.time()

                    self.fatigue_events = [event for event in self.fatigue_events if
                                           time.time() - event < self.FATIGUE_EVENT_WINDOW_SECONDS]
                    if len(self.fatigue_events) >= self.FATIGUE_EVENT_THRESHOLD:
                        fatigue_detected = True

                    # --- Drowsiness Event Detection ---
                    drowsiness_event = False
                    if is_mouth_open_wide:
                        if self.yawn_start_time is None:
                            self.yawn_start_time = time.time()
                        elif time.time() - self.yawn_start_time >= self.YAWN_DURATION_SECONDS:
                            if not self.is_yawning:
                                self.yawn_count += 1;
                                self.fatigue_events.append(time.time());
                                self.is_yawning = True
                            drowsiness_event = True
                    else:
                        self.yawn_start_time = None;
                        self.is_yawning = False
                    if not is_mouth_open_wide:
                        nose_tip_y = np.mean([p[1] for p in nose])
                        self.head_positions.append(nose_tip_y)
                        if len(self.head_positions) > self.HEAD_NOD_FRAME_WINDOW:
                            self.head_positions.pop(0)
                            if max(self.head_positions) - nose_tip_y > self.HEAD_NOD_THRESHOLD_PIXELS:
                                if not self.is_nodding:
                                    self.nod_count += 1;
                                    self.fatigue_events.append(time.time());
                                    self.is_nodding = True
                                drowsiness_event = True
                            else:
                                self.is_nodding = False

                    if is_eyes_closed and not drowsiness_event:
                        if self.eye_closure_start_time is None:
                            self.eye_closure_start_time = time.time()
                        elif time.time() - self.eye_closure_start_time >= self.EYE_CLOSURE_SECONDS_THRESHOLD:
                            drowsiness_event = True
                    else:
                        self.eye_closure_start_time = None

                    # --- NEW STATE MACHINE LOGIC ---
                    if self.current_state == self.STATE["ACTIVE"] and fatigue_detected:
                        self.current_state = self.STATE["FATIGUED"]
                    elif self.current_state == self.STATE["FATIGUED"]:
                        if drowsiness_event:
                            self.current_state = self.STATE["DROWSY"]
                        elif not fatigue_detected:
                            self.current_state = self.STATE["ACTIVE"]
                    elif self.current_state == self.STATE["DROWSY"] and not drowsiness_event:
                        self.current_state = self.STATE["FATIGUED"]

                    # Determine status text and color based on the final state
                    if self.current_state == self.STATE["DROWSY"]:
                        status_text = "DROWSINESS ALERT!";
                        color = (0, 0, 255)
                    elif self.current_state == self.STATE["FATIGUED"]:
                        status_text = "Fatigue Detected";
                        color = (0, 165, 255)
                    else:
                        status_text = "Active"

                    # --- END OF MODIFIED LOGIC ---

            cv2.putText(frame, f"Status: {status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            time_to_decide = self.FATIGUE_BLINK_RATE_SECONDS - (time.time() - self.blink_analysis_start_time)
            info_text_1 = f"Blink Rate: {self.display_blink_rate:.2f} bpm"
            info_text_2 = f"Fatigue Check In: {int(time_to_decide)}s"
            info_text_3 = f"Yawns: {self.yawn_count} | Nods: {self.nod_count}"

            cv2.putText(frame, info_text_1, (550, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, info_text_2, (550, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, info_text_3, (550, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.imshow("Drowsiness Detector", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"): break

        vs.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        detector = DrowsinessDetector()
        detector.run()
    except Exception as e:
        print(f"[FATAL ERROR] An unexpected error occurred: {e}")