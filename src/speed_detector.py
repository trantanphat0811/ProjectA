import cv2
import numpy as np
import logging
import torch
from ultralytics import YOLO
import supervision as sv
from datetime import datetime
import os
from pathlib import Path

from config import (
    MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    SPEED_LIMIT,
    FRAME_RATE,
    PIXEL_TO_METER_RATIO,
    VIOLATION_IMAGE_DIR
)

logger = logging.getLogger(__name__)

class SpeedDetector:
    def __init__(self):
        """Initialize the speed detector with YOLO model and ByteTrack tracker."""
        try:
            logger.info("Initializing SpeedDetector")
            self.model = YOLO(MODEL_PATH)
            self.tracker = sv.ByteTrack()
            self.track_history = {}  # Store position history for each track
            self.speed_measurements = {}  # Store speed measurements for each track
            self.violation_history = set()  # Store IDs of vehicles that have already been recorded for violation
            
            # Ensure violation image directory exists
            os.makedirs(VIOLATION_IMAGE_DIR, exist_ok=True)
            
            logger.info("SpeedDetector initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing SpeedDetector: {str(e)}")
            raise

    def calculate_speed(self, track_id, current_position, frame_number):
        """Calculate speed of a tracked object based on its position history."""
        try:
            if track_id not in self.track_history:
                self.track_history[track_id] = []
            
            self.track_history[track_id].append((current_position, frame_number))
            
            # Keep only last 30 frames of history
            if len(self.track_history[track_id]) > 30:
                self.track_history[track_id].pop(0)
            
            # Need at least 2 points to calculate speed
            if len(self.track_history[track_id]) < 2:
                return 0
            
            # Get positions and frames for calculation
            pos1, frame1 = self.track_history[track_id][0]
            pos2, frame2 = self.track_history[track_id][-1]
            
            # Calculate distance in meters
            distance = np.sqrt(
                (pos2[0] - pos1[0])**2 + 
                (pos2[1] - pos1[1])**2
            ) * PIXEL_TO_METER_RATIO
            
            # Calculate time in seconds
            time = (frame2 - frame1) / FRAME_RATE
            
            if time > 0:
                # Calculate speed in km/h
                speed = (distance / time) * 3.6
                
                # Update speed measurements
                if track_id not in self.speed_measurements:
                    self.speed_measurements[track_id] = []
                self.speed_measurements[track_id].append(speed)
                
                # Keep only last 10 measurements
                if len(self.speed_measurements[track_id]) > 10:
                    self.speed_measurements[track_id].pop(0)
                
                # Return average speed
                return np.mean(self.speed_measurements[track_id])
            
            return 0
            
        except Exception as e:
            logger.error(f"Error calculating speed: {str(e)}")
            return 0

    def save_violation_image(self, frame, track_id, speed, bbox, timestamp):
        """Save an image of the violation with details overlaid."""
        try:
            # Create a copy of the frame
            violation_image = frame.copy()
            
            # Draw bounding box
            cv2.rectangle(
                violation_image,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                (0, 0, 255),
                2
            )
            
            # Add violation details
            text_lines = [
                f"Vehicle ID: {track_id}",
                f"Speed: {speed:.1f} km/h",
                f"Speed Limit: {SPEED_LIMIT} km/h",
                f"Time: {timestamp}"
            ]
            
            y = int(bbox[1]) - 10
            for line in text_lines:
                y -= 20
                cv2.putText(
                    violation_image,
                    line,
                    (int(bbox[0]), y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2
                )
            
            # Save the image
            filename = f"violation_{track_id}_{timestamp.replace(':', '-')}.jpg"
            filepath = os.path.join(VIOLATION_IMAGE_DIR, filename)
            cv2.imwrite(filepath, violation_image)
            
            logger.info(f"Saved violation image to {filepath}")
            return filename
            
        except Exception as e:
            logger.error(f"Error saving violation image: {str(e)}")
            return None

    def process_frame(self, frame, frame_number):
        """Process a single frame and return annotated frame with detections."""
        try:
            # Run YOLO detection
            results = self.model(frame, conf=CONFIDENCE_THRESHOLD)[0]
            
            # Convert YOLO results to supervision Detections
            detections = sv.Detections.from_ultralytics(results)
            
            # Update tracks using ByteTracker
            if len(detections) > 0:
                tracks = self.tracker.track(detections)
            else:
                tracks = sv.Detections.empty()
            
            violations = []
            annotated_frame = frame.copy()
            
            # Process each track
            if len(tracks) > 0:
                for xyxy, confidence, class_id, tracker_id in zip(
                    tracks.xyxy, tracks.confidence, tracks.class_id, tracks.tracker_id
                ):
                    try:
                        # Calculate center point of bounding box
                        center_x = (xyxy[0] + xyxy[2]) / 2
                        center_y = (xyxy[1] + xyxy[3]) / 2
                        
                        # Calculate speed
                        speed = self.calculate_speed(
                            tracker_id,
                            (center_x, center_y),
                            frame_number
                        )
                        
                        # Draw bounding box and speed
                        color = (0, 0, 255) if speed > SPEED_LIMIT else (0, 255, 0)
                        cv2.rectangle(
                            annotated_frame,
                            (int(xyxy[0]), int(xyxy[1])),
                            (int(xyxy[2]), int(xyxy[3])),
                            color,
                            2
                        )
                        cv2.putText(
                            annotated_frame,
                            f"ID: {tracker_id} Speed: {speed:.1f} km/h",
                            (int(xyxy[0]), int(xyxy[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            2
                        )
                        
                        # Record violation if speed is above limit
                        if speed > SPEED_LIMIT and tracker_id not in self.violation_history:
                            timestamp = datetime.now().isoformat()
                            
                            # Save violation image
                            violation_image = self.save_violation_image(
                                frame,
                                tracker_id,
                                speed,
                                xyxy,
                                timestamp
                            )
                            
                            violations.append({
                                'track_id': tracker_id,
                                'speed': speed,
                                'frame_number': frame_number,
                                'timestamp': timestamp,
                                'confidence': float(confidence),
                                'image_file': violation_image
                            })
                            
                            # Add to violation history to avoid duplicate records
                            self.violation_history.add(tracker_id)
                            
                    except Exception as e:
                        logger.error(f"Error processing track {tracker_id}: {str(e)}")
                        continue
            
            return annotated_frame, violations
            
        except Exception as e:
            logger.error(f"Error processing frame {frame_number}: {str(e)}")
            return frame, []

    def reset(self):
        """Reset tracking history."""
        try:
            self.track_history.clear()
            self.speed_measurements.clear()
            self.violation_history.clear()
            logger.info("SpeedDetector reset successfully")
        except Exception as e:
            logger.error(f"Error resetting SpeedDetector: {str(e)}") 