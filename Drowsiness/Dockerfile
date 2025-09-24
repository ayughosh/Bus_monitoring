# Drowsiness/Dockerfile
#
# Dockerfile for the Drowsiness Detection application.
# It uses a multi-stage build for a smaller and more secure final image.

# --- Stage 1: Builder ---
# Use a full Python image to install dependencies and download the model.
FROM python:3.9-slim as builder

# Install system dependencies needed for dlib and opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    bzip2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the dlib facial landmark model
RUN wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 && \
    bzip2 -d shape_predictor_68_face_landmarks.dat.bz2


# --- Stage 2: Final Image ---
# Use a slim Python image for the final container
FROM python:3.9-slim

# Install only the necessary runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the installed Python packages from the builder stage
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages

# Copy your application code
COPY drowsiness_main.py .
COPY drowsiness_detector.py .
COPY config.py .

# Copy the downloaded dlib model from the builder stage
COPY --from=builder /app/shape_predictor_68_face_landmarks.dat .

# Set the command to run your application
CMD ["python", "drowsiness_main.py"]