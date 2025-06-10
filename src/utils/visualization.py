import cv2
import numpy as np
from src.core.config import SPEED_LIMITS, DISPLAY_CONFIG, CAMERA_CONFIG

def draw_results(frame, detections, stats=None):
    """Draw tracking results and speed on frame"""
    
    # Draw ROI
    height, width = frame.shape[:2]
    roi = CAMERA_CONFIG['roi']
    roi_points = np.array([
        [int(roi['x1'] * width), int(roi['y1'] * height)],
        [int(roi['x2'] * width), int(roi['y1'] * height)],
        [int(roi['x2'] * width), int(roi['y2'] * height)],
        [int(roi['x1'] * width), int(roi['y2'] * height)]
    ], np.int32)
    cv2.polylines(frame, [roi_points], True, DISPLAY_CONFIG['colors']['roi'], DISPLAY_CONFIG['line_thickness'])

    # Draw tracking and speed information
    for detection in detections:
        if not detection.get('tracker_id'):
            continue
            
        # Get box information
        x, y, w, h = map(int, detection['bbox'])
        
        # Get color based on violation status
        speed = detection.get('speed', 0)
        speed_limit = SPEED_LIMITS.get(detection['class_name'], 60)
        color = DISPLAY_CONFIG['colors']['violation'] if speed > speed_limit else DISPLAY_CONFIG['colors']['normal']
            
        # Draw bounding box
        if DISPLAY_CONFIG['show_tracking']:
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, DISPLAY_CONFIG['line_thickness'])
        
        # Draw information
        if DISPLAY_CONFIG['show_speed']:
            text = f"{detection['class_name']} #{detection['tracker_id']}"
            if speed > 0:
                text += f" - {speed:.1f} km/h"
                if speed > speed_limit and DISPLAY_CONFIG['show_violations']:
                    text += " [VIOLATION]"
            
            # Calculate text size for better positioning
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = DISPLAY_CONFIG['text_scale']
            thickness = DISPLAY_CONFIG['line_thickness']
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Draw text background
            cv2.rectangle(frame, 
                         (x, y - text_height - baseline - 5),
                         (x + text_width, y),
                         color, -1)
            
            # Draw text
            cv2.putText(frame, text,
                       (x, y - baseline - 5),
                       font, font_scale,
                       (255, 255, 255),  # White text
                       thickness)

    # Draw statistics if available
    if stats and DISPLAY_CONFIG.get('show_stats', True):
        y_pos = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        padding = 10
        
        # Draw stats background
        stats_height = len(stats) * 30
        cv2.rectangle(frame,
                     (0, 0),
                     (300, stats_height + padding * 2),
                     (0, 0, 0, 0.5),  # Semi-transparent black
                     -1)
        
        # Draw stats text
        for key, value in stats.items():
            if isinstance(value, float):
                text = f"{key}: {value:.1f}"
            else:
                text = f"{key}: {value}"
            cv2.putText(frame, text,
                       (padding, y_pos),
                       font, font_scale,
                       (255, 255, 255),  # White text
                       thickness)
            y_pos += 30

    return frame

def create_violation_image(frame, detection, speed):
    """Create a violation image with annotations"""
    x, y, w, h = map(int, detection['bbox'])
    
    # Add padding around the vehicle
    padding = 50
    y1 = max(0, y - padding)
    y2 = min(frame.shape[0], y + h + padding)
    x1 = max(0, x - padding)
    x2 = min(frame.shape[1], x + w + padding)
    
    # Crop the region
    violation_img = frame[y1:y2, x1:x2].copy()
    
    # Add violation information
    text = f"{detection['class_name']} - {speed:.1f} km/h"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    
    # Calculate text size for positioning
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Draw text background
    cv2.rectangle(violation_img,
                 (0, 0),
                 (text_width + 20, text_height + baseline + 10),
                 (0, 0, 0),
                 -1)
    
    # Draw text
    cv2.putText(violation_img, text,
                (10, text_height + 5),
                font, font_scale,
                (255, 255, 255),
                thickness)

    return violation_img 