import cv2
import time
import dlib
import numpy as np
from imutils import face_utils
from camera import Camera
from npu_inferencer import NPUInferencer
from utils import apply_clahe, draw_info
import config

# --- Drowsiness Logic Variables ---
DROWSY_CONSEC_FRAMES = 15
drowsy_counter = 0
current_status = "Active"


# --- End of Drowsiness Logic ---

def main():
    global drowsy_counter, current_status

    # Initialize components
    camera = Camera(config.CAMERA_URL)
    npu_inferencer = NPUInferencer()

    # --- Dlib and OpenCV Initializations ---
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcade_frontalface_default.xml')
    # Load the Dlib facial landmark predictor
    try:
        predictor = dlib.shape_predictor(config.DLIB_LANDMARK_MODEL)
    except Exception as e:
        print(f"[ERROR] Could not load Dlib predictor model: {e}")
        print(f"Make sure '{config.DLIB_LANDMARK_MODEL}' is in the project directory.")
        return
    # --- End of Initializations ---

    last_emotion = "Unknown"

    if not camera.start():
        return

    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            frame = cv2.resize(frame, (config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 1. Fast Face Detection (Haar Cascade)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)

            # Process only the first detected face for simplicity
            if len(faces) > 0:
                x, y, w, h = faces[0]

                # Create a dlib rectangle object from the Haar cascade's bounding box
                dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

                # 2. Precise Landmark Prediction (Dlib)
                shape = predictor(gray, dlib_rect)
                shape_np = face_utils.shape_to_np(shape)

                # 3. Align & Crop the face using landmarks
                # Find the bounding box of the landmarks to get a tight crop
                (lx, ly, lw, lh) = cv2.boundingRect(shape_np)
                # Add some padding
                padding = 10
                aligned_face = frame[max(0, ly - padding):ly + lh + padding, max(0, lx - padding):lx + lw + padding]

                # 4. TFLite Classification
                if aligned_face.size > 0:
                    last_emotion = npu_inferencer.infer(aligned_face)

                # --- Drowsiness Logic Integration ---
                mapped_emotion = config.EMOTION_MAPPING.get(last_emotion, 'attention')
                if mapped_emotion == 'drowsy':
                    drowsy_counter += 1
                    if drowsy_counter >= DROWSY_CONSEC_FRAMES:
                        current_status = "DROWSINESS ALERT!"
                else:
                    drowsy_counter = 0
                    current_status = "Active"
                # --- End of Logic Integration ---

                # Visualization: Draw landmarks and status
                for (sx, sy) in shape_np:
                    cv2.circle(frame, (sx, sy), 2, (0, 255, 0), -1)

                color = (0, 0, 255) if "ALERT" in current_status else (0, 255, 0)
                draw_info(frame, [x, y, w, h], current_status, color)

            cv2.imshow("Conductor Monitor", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()