FROM torizon/wayland-base-dev-imx8:4

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    libgtk-3-0 \
    v4l-utils \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download YOLO model
RUN mkdir -p models && \
    wget -q https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -P models/

# Copy application files
COPY bus_headcount_system.py .
COPY headcount_main.py .

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "headcount_main.py", "--web", "--camera", "0", "--host", "0.0.0.0", "--port", "5000"]