import cv2
import numpy as np
from ultralytics import YOLO
import os
import time
import json
import threading
import queue
from collections import deque
from flask import Flask, Response, render_template_string, jsonify
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizedHeadCountSystem:
    """
    Optimized head counting system with horizontal line for top-down movement
    """

    def __init__(self, camera_source=0, yolo_model_path="models/yolov8n.pt",
                zone_config=None, web_stream=True):
        """
        Initialize the optimized head count system
        """
        if not os.path.exists(yolo_model_path):
            logger.error(f"YOLO model not found at '{yolo_model_path}'")
            raise FileNotFoundError(f"YOLO model not found at '{yolo_model_path}'")

        # Performance settings
        self.PROCESS_EVERY_N_FRAMES = 2  # Process every 2nd frame for better performance
        self.RESIZE_WIDTH = 640  # Resize frame for faster processing
        self.RESIZE_HEIGHT = 480
        self.VIDEO_QUALITY = 70  # JPEG quality (1-100, lower = faster)

        # Initialize camera with optimized settings
        self.camera_source = camera_source
        self.cap = self._initialize_camera(camera_source)

        # Get original dimensions
        self.original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Zone configuration - HORIZONTAL LINE for top-down counting
        if zone_config:
            self.COUNTING_ZONE = zone_config
        else:
            # Default horizontal line in middle of frame
            self.COUNTING_ZONE = {
                'y': self.RESIZE_HEIGHT // 2,  # Horizontal line Y position
                'x1': 0,  # Line start X
                'x2': self.RESIZE_WIDTH,  # Line end X
                'threshold': 40,  # Pixels from line to trigger counting
                'direction': 'top_down'  # Entry from top, exit from bottom
            }

        # Load YOLO model
        logger.info(f"Loading YOLO model from {yolo_model_path}")
        self.model = YOLO(yolo_model_path)

        # Tracking data structures
        self.tracks = {}  # track_id -> TrackInfo
        self.counts = {'entry': 0, 'exit': 0, 'current': 0}
        self.track_history = deque(maxlen=500)  # Reduced history size

        # Web streaming
        self.web_stream = web_stream
        self.frame_queue = queue.Queue(maxsize=2)  # Smaller queue for less lag
        self.latest_frame = None
        self.stats_lock = threading.Lock()

        # Performance metrics
        self.fps = 0
        self.last_fps_time = time.time()
        self.frame_count = 0
        self.frame_skip_counter = 0

        logger.info("Optimized head count system initialized")

    def _initialize_camera(self, source, max_retries=3):
        """Initialize camera with optimized settings"""
        for attempt in range(max_retries):
            try:
                if isinstance(source, str) and source.startswith(('rtsp://', 'http://')):
                    cap = cv2.VideoCapture(source)
                elif isinstance(source, str) and source.isdigit():
                    cap = cv2.VideoCapture(int(source))
                else:
                    cap = cv2.VideoCapture(source)

                if cap.isOpened():
                    # Optimize camera settings for performance
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer
                    cap.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS for smoother streaming

                    # Set resolution if possible (some cameras ignore this)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

                    logger.info(f"Camera initialized on attempt {attempt + 1}")
                    return cap

            except Exception as e:
                logger.warning(f"Camera init attempt {attempt + 1} failed: {e}")

            if attempt < max_retries - 1:
                time.sleep(1)

        raise ConnectionError(f"Could not open camera source '{source}'")

    def _calculate_direction_vertical(self, track_id, current_y):
        """Calculate movement direction for vertical movement (top-down or bottom-up)"""
        if track_id not in self.tracks:
            return None

        track = self.tracks[track_id]
        if len(track['positions']) < 3:
            return None

        # Convert deque to list before slicing
        positions_list = list(track['positions'])

        # Get average Y positions
        start_positions = list(positions_list[:min(3, len(positions_list))])
        end_positions = list(positions_list[-min(3, len(positions_list)):])

        avg_start_y = np.mean([p[1] for p in start_positions])
        avg_end_y = np.mean([p[1] for p in end_positions])

        # Determine direction based on Y movement
        movement = avg_end_y - avg_start_y

        if self.COUNTING_ZONE['direction'] == 'top_down':
            # Entry from top (smaller Y), exit from bottom (larger Y)
            if movement > 15:  # Moving down (entry)
                return 'entry'
            elif movement < -15:  # Moving up (exit)
                return 'exit'
        else:  # bottom_up
            # Entry from bottom, exit from top
            if movement < -15:  # Moving up (entry)
                return 'entry'
            elif movement > 15:  # Moving down (exit)
                return 'exit'

        return None

    def _update_tracking(self, track_id, bbox, center):
        """
        MODIFIED: A simpler, more robust tracking and counting logic.
        """
        current_time = time.time()

        # Initialize new tracks (simplified, removed unused keys)
        if track_id not in self.tracks:
            self.tracks[track_id] = {
                'positions': deque(maxlen=20),
                'bbox': bbox,
                'last_seen': current_time,
                'counted': False
            }

        # Update track with current position
        track = self.tracks[track_id]
        track['positions'].append(center)
        track['bbox'] = bbox
        track['last_seen'] = current_time

        # --- REPLACED COUNTING LOGIC ---
        zone = self.COUNTING_ZONE
        line_y = zone['y']
        threshold = zone['threshold']

        # Check if the person is near the counting line and hasn't been counted yet
        if not track['counted'] and abs(center[1] - line_y) < threshold:

            # Determine the track's overall direction of movement
            # This requires the _calculate_direction_vertical function to exist in your class
            direction = self._calculate_direction_vertical(track_id, center[1])

            if direction:
                # If a clear direction is found, count the person and mark as counted
                self._count_person(track_id, direction)
                track['counted'] = True

    def _count_person(self, track_id, direction):
        """Count a person crossing the line (NO CHANGES NEEDED HERE)"""
        with self.stats_lock:
            if direction == 'entry':
                self.counts['entry'] += 1
                self.counts['current'] += 1
                logger.info(f"Person {track_id} ENTERED. Total inside: {self.counts['current']}")
            else:
                self.counts['exit'] += 1
                self.counts['current'] = max(0, self.counts['current'] - 1)
                logger.info(f"Person {track_id} EXITED. Total inside: {self.counts['current']}")

        # Add to history
        self.track_history.append({
            'track_id': track_id,
            'direction': direction,
            'timestamp': time.time()
        })

    def _cleanup_old_tracks(self):
        """Remove tracks that haven't been seen recently"""
        current_time = time.time()
        timeout = 3.0  # Increased timeout

        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            if current_time - track['last_seen'] > timeout:
                tracks_to_remove.append(track_id)

        for track_id in tracks_to_remove:
            del self.tracks[track_id]

    def _draw_visualization(self, frame, detections):
        """Draw horizontal counting zone and tracking information"""
        zone = self.COUNTING_ZONE

        # Draw horizontal counting line
        cv2.line(frame, (zone['x1'], zone['y']), (zone['x2'], zone['y']),
                (0, 255, 255), 3)

        # Draw threshold zones (above and below the line)
        cv2.line(frame, (zone['x1'], zone['y'] - zone['threshold']),
                (zone['x2'], zone['y'] - zone['threshold']), (255, 255, 0), 1)
        cv2.line(frame, (zone['x1'], zone['y'] + zone['threshold']),
                (zone['x2'], zone['y'] + zone['threshold']), (255, 255, 0), 1)

        # Draw zone labels
        cv2.putText(frame, "ENTRY", (zone['x1'], zone['y'] - zone['threshold'] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "EXIT", (zone['x1'], zone['y'] + zone['threshold'] + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Draw tracked persons
        for track_id, track in self.tracks.items():
            if len(track['positions']) > 0:
                center = track['positions'][-1]
                bbox = track['bbox']

                # Color based on position relative to line
                if center[1] < zone['y'] - zone['threshold']:
                    color = (0, 255, 0)  # Green above line
                elif center[1] > zone['y'] + zone['threshold']:
                    color = (0, 0, 255)  # Red below line
                else:
                    color = (255, 255, 0)  # Yellow on line

                # Draw bounding box
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

                # Draw ID
                cv2.putText(frame, f"ID:{track_id}", (bbox[0], bbox[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Draw center point
                cv2.circle(frame, tuple(map(int, center)), 4, color, -1)

                # Draw short trajectory
                if len(track['positions']) > 1:
                    points = list(track['positions'])[-10:]  # Last 10 points only
                    for i in range(1, len(points)):
                        cv2.line(frame, tuple(map(int, points[i - 1])),
                                 tuple(map(int, points[i])), color, 1)

        # Draw statistics panel
        self._draw_stats_panel(frame)
        return frame

    def _draw_stats_panel(self, frame):
        """Draw statistics panel"""
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (250, 130), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Draw statistics
        with self.stats_lock:
            stats = [
                f"Entries: {self.counts['entry']}",
                f"Exits: {self.counts['exit']}",
                f"Inside Bus: {self.counts['current']}",
                f"FPS: {self.fps:.1f}"
            ]

        y_offset = 35
        for stat in stats:
            cv2.putText(frame, stat, (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25

        # Timestamp
        timestamp = time.strftime("%H:%M:%S")
        cv2.putText(frame, timestamp, (frame.shape[1] - 100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _update_fps(self):
        """Calculate FPS"""
        self.frame_count += 1
        current_time = time.time()

        if current_time - self.last_fps_time > 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time

    def process_frame(self, frame):
        """Process frame with optimization"""
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (self.RESIZE_WIDTH, self.RESIZE_HEIGHT))

        # Skip frames for performance
        self.frame_skip_counter += 1
        if self.frame_skip_counter % self.PROCESS_EVERY_N_FRAMES == 0:
            # Run detection only on selected frames
            results = self.model.track(small_frame, persist=True, classes=[0],
                                    conf=0.3, tracker="bytetrack.yaml", verbose=False)

            if results[0].boxes is not None and results[0].boxes.id is not None:
                print(f"DEBUG: Detected {len(results[0].boxes.id)} objects in this frame.")
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)

                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = box

                    # Calculate center (use upper body)
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    center = (center_x, center_y)

                    # Update tracking
                    self._update_tracking(track_id, box, center)

            # Clean up old tracks
            if self.frame_skip_counter % 30 == 0:  # Clean every 30 frames
                self._cleanup_old_tracks()

        # Always draw visualization (even on skipped frames)
        visualized_frame = self._draw_visualization(small_frame.copy(), [])

        # Update FPS
        self._update_fps()

        return visualized_frame

    def run(self):
        """Main processing loop optimized for performance"""
        logger.info("Starting optimized head count processing...")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Failed to read frame")
                time.sleep(0.05)
                continue

            # Process frame
            processed_frame = self.process_frame(frame)

            # Update for web streaming
            if self.web_stream:
                self.latest_frame = processed_frame.copy()

                # Non-blocking queue update
                try:
                    # Clear old frames
                    while not self.frame_queue.empty():
                        try:
                            self.frame_queue.get_nowait()
                        except:
                            break
                    # Add new frame
                    self.frame_queue.put_nowait(processed_frame)
                except:
                    pass

            # Display locally
            # cv2.imshow("Bus Head Count System", processed_frame)

            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                with self.stats_lock:
                    self.counts = {'entry': 0, 'exit': 0, 'current': 0}
                logger.info("Counts reset")

        self.cleanup()

    def get_stats(self):
        """Get current statistics"""
        with self.stats_lock:
            return {
                'entries': self.counts['entry'],
                'exits': self.counts['exit'],
                'current': self.counts['current'],
                'fps': self.fps,
                'active_tracks': len(self.tracks)
            }

    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up...")
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


class OptimizedWebServer:
    """Optimized Flask web server for streaming"""

    def __init__(self, headcount_system, host='0.0.0.0', port=5000):
        self.app = Flask(__name__)
        self.headcount = headcount_system
        self.host = host
        self.port = port

        self.setup_routes()

    def setup_routes(self):
        """Setup Flask routes"""

        @self.app.route('/')
        def index():
            """Main page"""
            return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>Bus Head Count Monitor</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #1a1a1a;
            color: white;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            color: #00ff00;
            margin-bottom: 10px;
        }
        .info {
            text-align: center;
            color: #888;
            margin-bottom: 20px;
        }
        .video-container {
            text-align: center;
            margin: 20px 0;
            border: 2px solid #00ff00;
            border-radius: 10px;
            overflow: hidden;
            background: #000;
        }
        .video-container img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 0 auto;
        }
        .stats {
            display: flex;
            justify-content: space-around;
            margin: 20px 0;
            flex-wrap: wrap;
        }
        .stat-box {
            background: #2a2a2a;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            flex: 1;
            margin: 5px;
            min-width: 120px;
        }
        .stat-value {
            font-size: 32px;
            color: #00ff00;
            margin: 5px 0;
            font-weight: bold;
        }
        .stat-label {
            font-size: 12px;
            color: #888;
            text-transform: uppercase;
        }
        .controls {
            text-align: center;
            margin: 20px 0;
        }
        button {
            background: #00ff00;
            color: #000;
            border: none;
            padding: 10px 20px;
            margin: 0 5px;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
        }
        button:hover {
            background: #00dd00;
        }
        #timestamp {
            text-align: center;
            color: #888;
            margin: 10px 0;
        }
        .direction-info {
            text-align: center;
            color: #ffff00;
            margin: 10px 0;
        }
    </style>
    <script>
        function updateStats() {
            fetch('/stats')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('entries').textContent = data.entries;
                    document.getElementById('exits').textContent = data.exits;
                    document.getElementById('current').textContent = data.current;
                    document.getElementById('fps').textContent = data.fps.toFixed(1);
                    document.getElementById('timestamp').textContent = 
                        'Last updated: ' + new Date().toLocaleTimeString();
                });
        }

        function resetCounts() {
            if (confirm('Reset all counts to zero?')) {
                fetch('/reset', {method: 'POST'})
                    .then(() => updateStats());
            }
        }

        setInterval(updateStats, 1000);
        updateStats();
    </script>
</head>
<body>
    <div class="container">
        <h1>🚌 Bus Head Count Monitor</h1>
        <div class="info">Karnataka Government Transport Safety System</div>
        <div class="direction-info">↓ ENTRY (Top) | EXIT (Bottom) ↑</div>

        <div class="stats">
            <div class="stat-box">
                <div class="stat-label">Entries ↓</div>
                <div class="stat-value" id="entries">0</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Exits ↑</div>
                <div class="stat-value" id="exits">0</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Inside Bus</div>
                <div class="stat-value" id="current">0</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">FPS</div>
                <div class="stat-value" id="fps">0</div>
            </div>
        </div>

        <div class="video-container">
            <img src="/video_feed" />
        </div>

        <div class="controls">
            <button onclick="resetCounts()">Reset Counts</button>
        </div>

        <div id="timestamp"></div>
    </div>
</body>
</html>
            ''')

        @self.app.route('/video_feed')
        def video_feed():
            """Video streaming route"""
            return Response(self.generate_frames(),
                            mimetype='multipart/x-mixed-replace; boundary=frame')

        @self.app.route('/stats')
        def stats():
            """Get statistics"""
            return jsonify(self.headcount.get_stats())

        @self.app.route('/reset', methods=['POST'])
        def reset():
            """Reset counts"""
            with self.headcount.stats_lock:
                self.headcount.counts = {'entry': 0, 'exit': 0, 'current': 0}
            return jsonify({'status': 'reset'})

    def generate_frames(self):
        """Generate frames for streaming by pulling from the queue"""
        while True:
            try:
                # Block and wait for the next available frame from the queue.
                # The timeout prevents it from waiting forever if the main thread stops.
                frame = self.headcount.frame_queue.get(timeout=10)

                # Encode the frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    continue

                frame_bytes = buffer.tobytes()

                # Yield the frame in the multipart format
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            except queue.Empty:
                # If the queue is empty for 10 seconds, something might be wrong,
                # but we continue to keep the client connection open.
                logger.warning("Frame queue is empty, streaming is paused...")
                continue

    def run(self):
        """Start web server"""
        logger.info(f"Starting web server on http://{self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, threaded=True)


# Export the optimized versions with the original names for compatibility
ImprovedHeadCountSystem = OptimizedHeadCountSystem
HeadCountWebServer = OptimizedWebServer
HeadCountSystem = OptimizedHeadCountSystem  # Fallback compatibility

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Optimized Bus Head Count System')
    parser.add_argument('--camera', type=str, default='0', help='Camera source')
    parser.add_argument('--model', type=str, default='models/yolov8n.pt', help='YOLO model path')
    parser.add_argument('--web', action='store_true', help='Enable web interface')
    parser.add_argument('--port', type=int, default=5000, help='Web port')

    args = parser.parse_args()

    try:
        # Convert camera source
        camera_source = int(args.camera) if args.camera.isdigit() else args.camera

        # Initialize system
        headcount = OptimizedHeadCountSystem(
            camera_source=camera_source,
            yolo_model_path=args.model,
            web_stream=args.web
        )

        if args.web:
            # Start web server
            web_server = OptimizedWebServer(headcount, port=args.port)
            web_thread = threading.Thread(target=web_server.run)
            web_thread.daemon = True
            web_thread.start()

            time.sleep(2)
            print(f"\n✅ Web interface: http://0.0.0.0:{args.port}\n")

        # Run system
        headcount.run()

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback

        traceback.print_exc()