import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from collections import defaultdict
import logging
import os
from datetime import datetime
import sqlite3
from pathlib import Path
from .config import (
    SPEED_LIMITS, DISTANCE_REFERENCE, MIN_FRAMES_FOR_SPEED,
    MAX_FRAMES_FOR_SPEED, SPEED_SMOOTHING_FACTOR, CONFIDENCE_THRESHOLD,
    TRACKING_CONFIG, STORAGE_CONFIG, CAMERA_CONFIG, SPEED_CALCULATION,
    STATISTICS_CONFIG, MODEL_CONFIG, DISPLAY_CONFIG
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(STORAGE_CONFIG['upload_dir'] / 'speed_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SpeedDetector:
    def __init__(self, model=None):
        """Initialize Speed Detector"""
        # Initialize model if not provided
        if model is None:
            model_path = STORAGE_CONFIG['model_path']
            if not model_path.exists():
                logger.warning(f"Model not found at {model_path}. Downloading default model...")
                model = YOLO('yolov8n.pt')
            else:
                model = YOLO(str(model_path))
        self.model = model
        
        # Initialize tracker
        self.tracker = sv.ByteTrack()  # ByteTrack uses default parameters
        
        # Initialize vehicle trackers
        self.vehicle_trackers = {}
        
        # Initialize database
        self._init_database()
        
        logger.info("SpeedDetector initialized successfully")

    def _init_database(self):
        """Initialize database for storing violations"""
        try:
            conn = sqlite3.connect(STORAGE_CONFIG['database_path'])
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS violations
                        (id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT,
                        vehicle_type TEXT,
                        speed REAL,
                        speed_limit REAL,
                        image_path TEXT,
                        location_x REAL,
                        location_y REAL,
                        confidence REAL)''')
            conn.commit()
            conn.close()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise

    def process_frame(self, frame):
        """Process a frame and return detections"""
        try:
            # Run inference
            results = self.model(frame, verbose=False)[0]
            
            # Filter detections by class and confidence
            filtered_detections = []
            for box in results.boxes.data:
                x1, y1, x2, y2, conf, class_id = box[:6]
                class_name = results.names[int(class_id)]
                
                if (conf >= TRACKING_CONFIG['conf_thres'] and 
                    class_name in TRACKING_CONFIG['filter_classes']):
                    filtered_detections.append({
                        'bbox': [x1, y1, x2-x1, y2-y1],  # [x, y, w, h]
                        'confidence': conf,
                        'class_name': class_name,
                        'tracker_id': None  # Will be set by tracker
                    })
            
            # Update trackers
            if filtered_detections:
                detections_array = np.array([d['bbox'] for d in filtered_detections])
                tracks = self.tracker.update(
                    detections=sv.Detections.from_yolov8(results)
                )
                
                # Update detection objects with track IDs
                for track, detection in zip(tracks, filtered_detections):
                    detection['tracker_id'] = track.tracker_id
                    
                    # Update or create vehicle tracker
                    if track.tracker_id not in self.vehicle_trackers:
                        self.vehicle_trackers[track.tracker_id] = {
                            'positions': [],
                            'speeds': [],
                            'class_name': detection['class_name'],
                            'last_update': datetime.now(),
                            'violated': False
                        }
                    
                    tracker = self.vehicle_trackers[track.tracker_id]
                    tracker['positions'].append({
                        'bbox': detection['bbox'],
                        'timestamp': datetime.now(),
                        'confidence': detection['confidence']
                    })
                    
                    # Calculate speed if enough positions are recorded
                    if len(tracker['positions']) >= MIN_FRAMES_FOR_SPEED:
                        speed = self._calculate_speed(tracker['positions'])
                        tracker['speeds'].append(speed)
                        
                        # Check for violation
                        speed_limit = SPEED_LIMITS.get(detection['class_name'], 60)
                        if speed > speed_limit and not tracker['violated']:
                            tracker['violated'] = True
                            self._handle_violation(frame, detection, speed)
                
                # Clean up old trackers
                current_time = datetime.now()
                to_delete = []
                for track_id, tracker in self.vehicle_trackers.items():
                    time_diff = (current_time - tracker['last_update']).total_seconds()
                    if time_diff > STATISTICS_CONFIG['clear_inactive_after'] / CAMERA_CONFIG['fps']:
                        to_delete.append(track_id)
                
                for track_id in to_delete:
                    del self.vehicle_trackers[track_id]
            
            return filtered_detections
            
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return []
        
    def _calculate_speed(self, positions):
        """Calculate speed from position history"""
        if len(positions) < 2:
            return 0
            
        # Get last two positions
        pos1 = positions[-2]
        pos2 = positions[-1]
        
        # Calculate time difference in seconds
        time_diff = (pos2['timestamp'] - pos1['timestamp']).total_seconds()
        if time_diff <= 0:
            return 0
            
        # Calculate distance in pixels
        x1, y1 = pos1['bbox'][0] + pos1['bbox'][2]/2, pos1['bbox'][1] + pos1['bbox'][3]/2
        x2, y2 = pos2['bbox'][0] + pos2['bbox'][2]/2, pos2['bbox'][1] + pos2['bbox'][3]/2
        pixel_distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        
        # Convert pixel distance to meters using reference distance
        meters = pixel_distance * DISTANCE_REFERENCE / CAMERA_CONFIG['width']
        
        # Calculate speed in km/h
        speed = (meters / time_diff) * 3.6
        
        # Apply smoothing
        if positions[-1].get('speed'):
            speed = speed * (1-SPEED_SMOOTHING_FACTOR) + positions[-1]['speed'] * SPEED_SMOOTHING_FACTOR
            
        return speed
        
    def _handle_violation(self, frame, detection, speed):
        """Handle speed limit violation"""
        try:
            # Create violation record
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_name = f"violation_{timestamp}_{detection['class_name']}.jpg"
            image_path = STORAGE_CONFIG['violations_dir'] / image_name
            
            # Save violation image
            x, y, w, h = detection['bbox']
            cv2.imwrite(str(image_path), frame[int(y):int(y+h), int(x):int(x+w)])
            
            # Save to database
            conn = sqlite3.connect(STORAGE_CONFIG['database_path'])
            c = conn.cursor()
            c.execute('''INSERT INTO violations 
                        (timestamp, vehicle_type, speed, speed_limit, image_path, 
                         location_x, location_y, confidence)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                     (timestamp, detection['class_name'], speed,
                      SPEED_LIMITS.get(detection['class_name'], 60),
                      str(image_path), x, y, detection['confidence']))
            conn.commit()
            conn.close()
            
            logger.info(f"Violation recorded: {detection['class_name']} at {speed:.1f} km/h")
            
        except Exception as e:
            logger.error(f"Error handling violation: {str(e)}")

    def get_statistics(self):
        """Get detection statistics"""
        try:
            stats = {
                'total_vehicles': len(self.vehicle_trackers),
                'violations': 0,
                'average_speed': 0,
                'class_counts': defaultdict(int),
                'speed_distribution': defaultdict(list)
            }
            
            total_speed = 0
            speed_count = 0
            
            for tracker in self.vehicle_trackers.values():
                stats['class_counts'][tracker['class_name']] += 1
                if tracker['violated']:
                    stats['violations'] += 1
                if tracker['speeds']:
                    avg_speed = sum(tracker['speeds']) / len(tracker['speeds'])
                    total_speed += avg_speed
                    speed_count += 1
                    stats['speed_distribution'][tracker['class_name']].append(avg_speed)
            
            if speed_count > 0:
                stats['average_speed'] = total_speed / speed_count
                
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {str(e)}")
            return {}

if __name__ == "__main__":
    detector = SpeedDetector()
    detector.process_video('sample_video.mp4')