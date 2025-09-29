FROM torizon/wayland-base-dev-imx8:4

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    libatlas-base-dev \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-python-dev \
    libboost-thread-dev \
    libatlas-base-dev \
    libopencv-dev \
    python3-opencv \
    wget \
    bzip2 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3-full \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
# --- Add a 2GB swap file to provide enough memory for compilation ---
RUN fallocate -l 2G /swapfile && \
    chmod 600 /swapfile && \
    mkswap /swapfile && \
    swapon /swapfile

# Run pip install. This will now compile dlib, which will be slow.
RUN pip3 install -r requirements.txt --break-system-packages

# --- Turn off swap and remove the temporary file ---
RUN swapoff /swapfile && \
    rm /swapfile

COPY shape_predictor_68_face_landmarks.dat .
COPY drowsiness_main.py .
COPY drowsiness_detector.py .
COPY config.py .

CMD ["python", "drowsiness_main.py"]