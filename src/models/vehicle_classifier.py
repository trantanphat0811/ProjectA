import cv2
from ultralytics import YOLO
from collections import Counter
import sqlite3
from datetime import datetime
from data_preparer import DATABASE_CONFIG, MODEL_CONFIG, VEHICLE_CLASSES, LOGGING_CONFIG  # Sửa import
import logging

# Cấu hình logging
logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)

def init_db():
    """Khởi tạo cơ sở dữ liệu SQLite."""
    conn = sqlite3.connect(DATABASE_CONFIG['path'])
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            class_name TEXT,
            confidence REAL,
            bbox_x REAL,
            bbox_y REAL,
            bbox_w REAL,
            bbox_h REAL
        )
    ''')
    conn.commit()
    conn.close()
    logger.info("Cơ sở dữ liệu đã được khởi tạo")

def save_detection(class_name, confidence, bbox):
    """Lưu thông tin phát hiện vào cơ sở dữ liệu."""
    conn = sqlite3.connect(DATABASE_CONFIG['path'])
    c = conn.cursor()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    c.execute('''
        INSERT INTO detections (timestamp, class_name, confidence, bbox_x, bbox_y, bbox_w, bbox_h)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (timestamp, class_name, confidence, *bbox))
    conn.commit()
    conn.close()
    logger.info(f"Đã lưu phát hiện: {class_name}, confidence: {confidence}")

def classify_vehicles(video_path, model_path='yolov8n.pt'):
    """Phân loại phương tiện trong video và lưu vào cơ sở dữ liệu."""
    init_db()  # Khởi tạo cơ sở dữ liệu
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    vehicle_types = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, conf=MODEL_CONFIG['confidence_threshold'], iou=MODEL_CONFIG['iou_threshold'])
        for det in results[0].boxes:
            if det.conf > MODEL_CONFIG['confidence_threshold']:
                cls = int(det.cls)
                label = model.names[cls]
                if label in VEHICLE_CLASSES.values():
                    vehicle_types.append(label)
                    # Lưu vào cơ sở dữ liệu
                    bbox = det.xywh[0].tolist()  # [x_center, y_center, width, height]
                    save_detection(label, det.conf.item(), bbox)
        cv2.imshow('Vehicle Classification', results[0].plot())
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    counts = Counter(vehicle_types)
    print("Vehicle counts:", dict(counts))
    logger.info(f"Vehicle counts: {dict(counts)}")
    return counts

if __name__ == "__main__":
    classify_vehicles('sample_video.mp4')