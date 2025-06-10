import cv2
import numpy as np
from ultralytics import YOLO
import sqlite3
from datetime import datetime
import os

class SpeedDetector:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)
        self.tracked_objects = {}
        self.speed_threshold = 50  # km/h
        
    def calculate_speed(self, prev_pos, curr_pos, fps):
        # Convert pixel distance to meters (assuming 1 pixel = 0.1 meters)
        pixel_to_meter = 0.1
        distance = np.sqrt(
            (curr_pos[0] - prev_pos[0])**2 + 
            (curr_pos[1] - prev_pos[1])**2
        ) * pixel_to_meter
        
        # Calculate speed in km/h
        time = 1/fps  # seconds
        speed = (distance / time) * 3.6  # Convert m/s to km/h
        return speed
        
    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Database connection
        conn = sqlite3.connect('detections.db')
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS videos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                upload_date DATETIME,
                processed_date DATETIME,
                total_vehicles INTEGER,
                violations INTEGER
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER,
                timestamp DATETIME,
                speed FLOAT,
                vehicle_type TEXT,
                frame_number INTEGER,
                FOREIGN KEY (video_id) REFERENCES videos (id)
            )
        ''')
        
        # Insert video record
        cursor.execute('''
            INSERT INTO videos (filename, upload_date, processed_date)
            VALUES (?, ?, ?)
        ''', (os.path.basename(video_path), datetime.now(), None))
        video_id = cursor.lastrowid
        
        total_vehicles = 0
        violations = 0
        frame_count = 0
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            results = self.model.track(frame, persist=True)
            
            if results and results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    center = (float(x), float(y))
                    
                    if track_id in self.tracked_objects:
                        prev_center, prev_frame = self.tracked_objects[track_id]
                        frames_elapsed = frame_count - prev_frame
                        
                        if frames_elapsed > 0:
                            speed = self.calculate_speed(prev_center, center, fps/frames_elapsed)
                            
                            if speed > self.speed_threshold:
                                violations += 1
                                cursor.execute('''
                                    INSERT INTO violations (video_id, timestamp, speed, vehicle_type, frame_number)
                                    VALUES (?, ?, ?, ?, ?)
                                ''', (video_id, datetime.now(), speed, 'vehicle', frame_count))
                    
                    self.tracked_objects[track_id] = (center, frame_count)
                    total_vehicles = max(total_vehicles, len(self.tracked_objects))
            
            frame_count += 1
        
        # Update video record
        cursor.execute('''
            UPDATE videos 
            SET processed_date = ?, total_vehicles = ?, violations = ?
            WHERE id = ?
        ''', (datetime.now(), total_vehicles, violations, video_id))
        
        conn.commit()
        conn.close()
        cap.release()
        
        return {
            'video_id': video_id,
            'total_vehicles': total_vehicles,
            'violations': violations
        } 