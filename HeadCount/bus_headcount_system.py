# HeadCount/bus_headcount_system.py
"""
Improved Bus Head Count System with proper entry/exit detection
and web streaming capability
"""

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


class ImprovedHeadCountSystem:
    """
    Robust head counting system with improved tracking and entry/exit logic
    """

    def __init__(self, camera_source=0, yolo_model_path="models/yolov8n.pt",
                 zone_config=None, web_stream=True):
        """
        Initialize the improved head count system

        Args:
            camera_source: Camera index or RTSP URL
            yolo_model_path: Path to YOLO model
            zone_config: Custom zone configuration
            web_stream: Enable web streaming
        """
        # Validate model path
        if not os.path.exists(yolo_model_path):
            logger.error(f"YOLO model not found at '{yolo_model_path}'")
            raise FileNotFoundError(f"YOLO model not found at '{yolo_model_path}'")

        # Initialize camera with retry logic
        self.camera_source = camera_source
        self.cap = self._initialize_camera(camera_source)

        # Frame dimensions
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Zone configuration
        if zone_config:
            self.COUNTING_ZONE = zone_config
        else:
            # Default zone - vertical line in middle of frame
            zone_y = self.frame_width // 2
            self.COUNTING_ZONE = {
                'y': zone_y,
                'x1': 50,
                'x2': self.frame_height - 50,
                'threshold': 30  # pixels from line to trigger counting
            }

        # Load YOLO model
        logger.info(f"Loading YOLO model from {yolo_model_path}")
        self.model = YOLO(yolo_model_path)

        # Tracking data structures
        self.tracks = {}  # track_id -> TrackInfo
        self.counts = {'entry': 0, 'exit': 0, 'current': 0}
        self.track_history = deque(maxlen=1000)  # Keep history for debugging

        # Web streaming
        self.web_stream = web_stream
        self.frame_queue = queue.Queue(maxsize=5)
        self.latest_frame = None
        self.stats_lock = threading.Lock()

        # Performance metrics
        self.fps = 0
        self.last_fps_time = time.time()
        self.frame_count = 0

        logger.info("Head count system initialized successfully")

    def _initialize_camera(self, source, max_retries=5):
        """Initialize camera with retry logic"""
        for attempt in range(max_retries):
            try:
                # Try different approaches based on source type
                if isinstance(source, str) and source.startswith(('rtsp://', 'http://')):
                    # Network camera
                    cap = cv2.VideoCapture(source)
                elif isinstance(source, str) and source.isdigit():
                    # Camera index as string
                    cap = cv2.VideoCapture(int(source))
                else:
                    # Direct camera index or file
                    cap = cv2.VideoCapture(source)

                if cap.isOpened():
                    # Set camera properties for better performance
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    # cap.set(cv2.CAP_PROP_FPS, 30)
                    logger.info(f"Camera initialized successfully on attempt {attempt + 1}")
                    return cap

            except Exception as e:
                logger.warning(f"Camera init attempt {attempt + 1} failed: {e}")

            if attempt < max_retries - 1:
                time.sleep(2)  # Wait before retry

        raise ConnectionError(f"Could not open camera source '{source}' after {max_retries} attempts")

    def _calculate_direction(self, track_id, current_x):
        """Calculate movement direction for a track"""
        if track_id not in self.tracks:
            return None

        track = self.tracks[track_id]
        if len(track['positions']) < 2:
            return None

        # Get average of first and last positions
        start_positions = track['positions'][:min(3, len(track['positions']))]
        end_positions = track['positions'][-min(3, len(track['positions'])):]

        avg_start_y = np.mean([p[1] for p in start_positions])
        avg_end_y = np.mean([p[1] for p in end_positions])

        # Determine direction
        if avg_end_y > avg_start_y + 10:  # Moving DOWN (entry)
            return 'entry'
        elif avg_end_y < avg_start_y - 10:  # Moving UP (exit)
            return 'exit'
        return None

    def _update_tracking(self, track_id, bbox, center):
        """Update tracking information for a person"""
        current_time = time.time()

        if track_id not in self.tracks:
            # New track
            self.tracks[track_id] = {
                'positions': deque(maxlen=30),
                'bbox': bbox,
                'first_seen': current_time,
                'last_seen': current_time,
                'counted': False,
                'direction': None,
                'crossed_line': False
            }

        track = self.tracks[track_id]
        track['positions'].append(center)
        track['bbox'] = bbox
        track['last_seen'] = current_time

        # Check if crossing the counting line
        zone = self.COUNTING_ZONE
        line_x = zone['y']
        threshold = zone['threshold']

        # Check if person is near the line
        if abs(center[1] - line_x) < threshold:
            if not track['crossed_line']:
                # Determine direction when crossing
                direction = self._calculate_direction(track_id, center[1])

                if direction and not track['counted']:
                    track['direction'] = direction
                    track['counted'] = True
                    track['crossed_line'] = True

                    # Update counts
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
                        'timestamp': current_time
                    })
        else:
            # Reset crossed_line flag when person moves away from line
            if track['crossed_line'] and abs(center[0] - line_x) > threshold * 2:
                track['crossed_line'] = False
                track['counted'] = False  # Allow re-counting if they cross again

    def _cleanup_old_tracks(self):
        """Remove tracks that haven't been seen recently"""
        current_time = time.time()
        timeout = 2.0  # seconds

        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            if current_time - track['last_seen'] > timeout:
                tracks_to_remove.append(track_id)

        for track_id in tracks_to_remove:
            del self.tracks[track_id]

    def _draw_visualization(self, frame, detections):
        """Draw counting zone and tracking information"""
        zone = self.COUNTING_ZONE

        # Draw counting line
        cv2.line(frame, (zone['y'], zone['x1']), (zone['y'], zone['x2']),
                 (0, 255, 255), 3)

        # Draw threshold zones
        cv2.line(frame, (zone['y'] - zone['threshold'], zone['x1']),
                 (zone['y'] - zone['threshold'], zone['x2']), (255, 255, 0), 1)
        cv2.line(frame, (zone['y'] + zone['threshold'], zone['x1']),
                 (zone['y'] + zone['threshold'], zone['x2']), (255, 255, 0), 1)

        # Draw zone label
        cv2.putText(frame, "ENTRY/EXIT ZONE", (zone['y'] - 60, zone['x1'] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Draw tracked persons
        for track_id, track in self.tracks.items():
            if len(track['positions']) > 0:
                # Get latest position
                center = track['positions'][-1]
                bbox = track['bbox']

                # Choose color based on state
                if track['counted']:
                    if track['direction'] == 'entry':
                        color = (0, 255, 0)  # Green for entry
                    else:
                        color = (0, 0, 255)  # Red for exit
                else:
                    color = (255, 255, 0)  # Yellow for tracking

                # Draw bounding box
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

                # Draw ID and center point
                cv2.putText(frame, f"ID:{track_id}", (bbox[0], bbox[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.circle(frame, tuple(map(int, center)), 4, color, -1)

                # Draw trajectory
                if len(track['positions']) > 1:
                    points = np.array([tuple(map(int, p)) for p in track['positions']],
                                      dtype=np.int32)
                    cv2.polylines(frame, [points], False, color, 1)

        # Draw statistics panel
        self._draw_stats_panel(frame)

        return frame

    def _draw_stats_panel(self, frame):
        """Draw statistics panel on frame"""
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (250, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Draw statistics
        with self.stats_lock:
            stats = [
                f"Entries: {self.counts['entry']}",
                f"Exits: {self.counts['exit']}",
                f"Current: {self.counts['current']}",
                f"FPS: {self.fps:.1f}"
            ]

        y_offset = 40
        for stat in stats:
            cv2.putText(frame, stat, (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30

        # Add timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _update_fps(self):
        """Calculate and update FPS"""
        self.frame_count += 1
        current_time = time.time()

        if current_time - self.last_fps_time > 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time

    def process_frame(self, frame):
        """Process a single frame for head counting"""
        # Run YOLO detection with tracking
        results = self.model.track(frame, persist=True, classes=[0],
                                   conf=0.4, tracker="bytetrack.yaml")

        detections = []

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box

                # Calculate center of person (use upper body for better tracking)
                center_x = (x1 + x2) // 2
                center_y = y1 + (y2 - y1) // 3  # Upper third of bounding box
                center = (center_x, center_y)

                # Update tracking
                self._update_tracking(track_id, box, center)

                detections.append({
                    'track_id': track_id,
                    'bbox': box,
                    'center': center
                })

        # Clean up old tracks
        self._cleanup_old_tracks()

        # Draw visualization
        visualized_frame = self._draw_visualization(frame.copy(), detections)

        # Update FPS
        self._update_fps()

        return visualized_frame

    def run(self):
        """Main processing loop"""
        logger.info("Starting head count processing...")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Failed to read frame, retrying...")
                time.sleep(0.1)
                continue

            # Process frame
            processed_frame = self.process_frame(frame)

            # Update latest frame for web streaming
            if self.web_stream:
                self.latest_frame = processed_frame.copy()

                # Try to put in queue, drop old frames if full
                try:
                    self.frame_queue.put_nowait(processed_frame)
                except queue.Full:
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(processed_frame)
                    except:
                        pass

            # Display locally (optional)
            # cv2.imshow("Bus Head Count System", processed_frame)

            # Check for quit
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            # break

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
        logger.info("Cleaning up resources...")
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


# Web server for streaming
class HeadCountWebServer:
    """Flask web server for streaming head count video"""

    def __init__(self, headcount_system, host='0.0.0.0', port=5000):
        self.app = Flask(__name__)
        self.headcount = headcount_system
        self.host = host
        self.port = port

        # Setup routes
        self.setup_routes()

    def setup_routes(self):
        """Setup Flask routes"""

        @self.app.route('/')
        def index():
            """Main page with video stream"""
            return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>Bus Head Count Monitor</title>
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
        }
        .video-container {
            text-align: center;
            margin: 20px 0;
            border: 2px solid #00ff00;
            border-radius: 10px;
            overflow: hidden;
        }
        .stats {
            display: flex;
            justify-content: space-around;
            margin: 20px 0;
        }
        .stat-box {
            background: #2a2a2a;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            flex: 1;
            margin: 0 10px;
        }
        .stat-value {
            font-size: 36px;
            color: #00ff00;
            margin: 10px 0;
        }
        .stat-label {
            font-size: 14px;
            color: #888;
        }
        #timestamp {
            text-align: center;
            color: #888;
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
                        'Last updated: ' + new Date().toLocaleString();
                });
        }

        setInterval(updateStats, 1000);
        updateStats();
    </script>
</head>
<body>
    <div class="container">
        <h1>🚌 Bus Head Count Monitoring System</h1>

        <div class="stats">
            <div class="stat-box">
                <div class="stat-label">ENTRIES</div>
                <div class="stat-value" id="entries">0</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">EXITS</div>
                <div class="stat-value" id="exits">0</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">CURRENT</div>
                <div class="stat-value" id="current">0</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">FPS</div>
                <div class="stat-value" id="fps">0</div>
            </div>
        </div>

        <div class="video-container">
            <img src="/video_feed" width="100%" />
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
            """Get current statistics"""
            return jsonify(self.headcount.get_stats())

    def generate_frames(self):
        """Generate frames for streaming"""
        while True:
            if self.headcount.latest_frame is not None:
                # Encode frame
                ret, buffer = cv2.imencode('.jpg', self.headcount.latest_frame)
                if ret:
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.033)  # ~30 FPS

    def run(self):
        """Start the web server"""
        logger.info(f"Starting web server on http://{self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, threaded=True)


# Main entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Bus Head Count System')
    parser.add_argument('--camera', type=str, default='0',
                        help='Camera source (index or URL)')
    parser.add_argument('--model', type=str, default='models/yolov8n.pt',
                        help='Path to YOLO model')
    parser.add_argument('--web', action='store_true',
                        help='Enable web interface')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Web server host')
    parser.add_argument('--port', type=int, default=5000,
                        help='Web server port')

    args = parser.parse_args()

    try:
        # Initialize head count system
        headcount = ImprovedHeadCountSystem(
            camera_source=args.camera if not args.camera.isdigit() else int(args.camera),
            yolo_model_path=args.model,
            web_stream=args.web
        )

        if args.web:
            # Start web server in separate thread
            web_server = HeadCountWebServer(headcount, args.host, args.port)
            web_thread = threading.Thread(target=web_server.run)
            web_thread.daemon = True
            web_thread.start()

            # Give server time to start
            time.sleep(2)
            print(f"\n✅ Web interface available at http://{args.host}:{args.port}\n")

        # Run head counting
        headcount.run()

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback

        traceback.print_exc()