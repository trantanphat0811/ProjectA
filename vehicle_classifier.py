cat <<EOL > vehicle_classifier.py
# vehicle_classifier.py - Classify vehicles using YOLO
from ultralytics import YOLO
import cv2
from collections import Counter

def classify_vehicles(video_path, model_path='yolov8n.pt'):
    """Classify vehicles in a video (e.g., car, truck, bus)."""
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    vehicle_types = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        for det in results[0].boxes:
            if det.conf > 0.5:
                cls = int(det.cls)
                label = model.names[cls]
                if label in ['car', 'truck', 'bus']:
                    vehicle_types.append(label)
        cv2.imshow('Vehicle Classification', results[0].plot())
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    counts = Counter(vehicle_types)
    print("Vehicle counts:", dict(counts))
    return counts

if __name__ == "__main__":
    classify_vehicles('sample_video.mp4')
EOL