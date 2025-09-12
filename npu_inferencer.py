# import cv2
# import numpy as np
# import platform  # Import the platformatform module to check the system architecture
# from config import EMOTION_MODEL, EMOTION_LABELS, INPUT_SIZE
#
# # --- Conditional Import based on System Architecture ---
# # This block checks if the machine is ARM64 (like the Verdin board) or x86_64 (your PC)
# IS_ARM64 = platform.machine() in ('aarch64', 'arm64')
#
# if IS_ARM64:
#     # On the ARM64 device, import the lightweight tflite_runtime
#     print("✅ Detected ARM64 architecture. Using tflite-runtime.")
#     from tflite_runtime.interpreter import Interpreter, load_delegate
# else:
#     # On your PC, import the full TensorFlow library
#     print("✅ Detected x86_64 architecture. Using full TensorFlow.")
#     import tensorflow as tf
#
#     # Create aliases to match the tflite_runtime structure
#     Interpreter = tf.lite.Interpreter
#     load_delegate = tf.lite.load_delegate
#
#
# # --- End of Conditional Import ---
#
#
# class NPUInferencer:
#     def __init__(self):
#         """
#         Initializes the TFLite interpreter, attempting to use the NPU delegate on ARM
#         and falling back to the CPU on all platforms if the delegate is not available.
#         """
#         try:
#             # This will only succeed on the Verdin board where the delegate library exists
#             self.interpreter = Interpreter(
#                 model_path=EMOTION_MODEL,
#                 experimental_delegates=[load_delegate('libvx_delegate.so')]
#             )
#             print("✅ Successfully loaded NPU delegate for hardware acceleration.")
#         except (ValueError, OSError):
#             # This will be the default path on your PC, or on ARM if the delegate is missing
#             print("⚠️ NPU delegate not found. Falling back to CPU for inference.")
#             self.interpreter = Interpreter(model_path=EMOTION_MODEL)
#
#         self.interpreter.allocate_tensors()
#         self.input_details = self.interpreter.get_input_details()
#         self.output_details = self.interpreter.get_output_details()
#         print("NPU Inferencer initialized.")
#
#     def infer(self, face_image):
#         """Performs inference on a single face image."""
#         # Preprocess the image to match the model's input requirements
#         img_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
#         img_resized = cv2.resize(img_gray, INPUT_SIZE)
#
#         # Expand dimensions to create a batch of 1 and add a channel dimension
#         img_expanded = np.expand_dims(img_resized, axis=0)
#         img_expanded = np.expand_dims(img_expanded, axis=-1).astype(np.float32)
#
#         # Set the tensor and invoke the interpreter
#         self.interpreter.set_tensor(self.input_details[0]['index'], img_expanded)
#         self.interpreter.invoke()
#
#         # Get the results
#         output = self.interpreter.get_tensor(self.output_details[0]['index'])
#         emotion_id = np.argmax(output)
#
#         return EMOTION_LABELS[emotion_id]