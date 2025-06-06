# traffic_density.py - Estimate traffic density using YOLO
from ultralytics import YOLO
import cv2
import numpy as np

def estimate_traffic_density(video_path, model_path='yolov8n.pt'):
    """Estimate vehicle density in a video."""
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    vehicle_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        vehicle_count += len([det for det in results[0].boxes if det.conf > 0.5])
        cv2.imshow('Traffic Density', results[0].plot())
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Total vehicles detected: {vehicle_count}")
    return vehicle_count

if __name__ == "__main__":
    estimate_traffic_density('sample_video.mp4')