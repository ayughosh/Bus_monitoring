# --- Model and Path Settings ---
# Path to the TFLite model for emotion detection
#EMOTION_MODEL = "models/emotion_model.tflite"

# Path to the Dlib facial landmark model
DLIB_LANDMARK_MODEL = "shape_predictor_68_face_landmarks.dat" # <<< ADD THIS LINE

# List of emotion labels in the order your model was trained
#EMOTION_LABELS = ['angry', 'drowsy', 'happy', 'neutral', 'sleepy', 'surprised']

# The input size (height, width) required by your TFLite model
# INPUT_SIZE = (48, 48)


# --- Real-Time Processing Parameters ---
# Your IP camera's RTSP streaming URL.
CAMERA_URL = 0 # Using webcam for testing


# --- Calibration Settings ---
# How many frames to use for calibration
CALIBRATION_FRAMES = 60

# --- Drowsiness Detection Logic ---
# EAR below this percentage of calibrated baseline triggers "closed" state
EAR_PERCENTAGE_THRESHOLD = 0.85

# How many consecutive frames of "closed" + "still" to trigger an alert
DROWSY_CONSEC_FRAMES = 25

# How much the head can move (in pixels) between frames to be considered "still"
HEAD_STILLNESS_THRESHOLD = 3.5

# --- Yawn Detection Logic ---
# MAR above this threshold triggers an immediate "Yawn" alert
YAWN_MAR_THRESHOLD = 0.9

# Number of frames to add to the drowsy counter when a yawn is detected
YAWN_FRAMES_BONUS = 20

# -- Face Detection Persistence --
FACE_LOST_THRESHOLD = 25

# -- Drowsiness Method 1: Time-based Eye Closure --
EYE_CLOSURE_SECONDS_THRESHOLD = 2.0


# # Display resolution for the output window
# DISPLAY_WIDTH = 800
# DISPLAY_HEIGHT = 600
#
# # Perform NPU inference on every Nth frame to save resources
# INFERENCE_INTERVAL = 1 # Set to 1 for smoother landmark detection
#
#
# # --- Logic and Mapping ---
# # Map the model's output labels to logical states for your application
# EMOTION_MAPPING = {
#     'drowsy': 'drowsy',
#     'sleepy': 'drowsy',
#     'angry': 'anger',
#     'happy': 'attention',
#     'neutral': 'attention',
#     'surprised': 'attention'
# }