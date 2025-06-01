import cv2
from ultralytics import YOLO
import numpy as np
from data_preparer import MODEL_CONFIG, VIDEO_CONFIG, LOGGING_CONFIG
import logging

# Cấu hình logging
logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)

class SpeedDetector:
    def __init__(self, model_path=None):
        self.model = YOLO(model_path or MODEL_CONFIG['model_name'])
        self.prev_positions = {}  # Lưu vị trí trước đó của các phương tiện
        self.frame_count = 0
        self.fps = VIDEO_CONFIG.get('fps', 30)  # Giả định FPS mặc định
        self.scale_factor = VIDEO_CONFIG.get('scale_factor', 0.01)  # Hệ số tỷ lệ (m/pixel)

    def calculate_speed(self, bbox, current_frame_id):
        """Tính tốc độ dựa trên vị trí trước và hiện tại."""
        obj_id = hash(str(bbox))  # ID tạm thời dựa trên bounding box
        x, y, w, h = bbox
        center = (x + w // 2, y + h // 2)

        if obj_id in self.prev_positions:
            prev_center, prev_frame_id = self.prev_positions[obj_id]
            time_diff = (current_frame_id - prev_frame_id) / self.fps  # Thời gian (giây)
            if time_diff > 0:
                dist_pixels = np.sqrt((center[0] - prev_center[0])**2 + (center[1] - prev_center[1])**2)
                dist_meters = dist_pixels * self.scale_factor  # Chuyển đổi sang mét
                speed_ms = dist_meters / time_diff  # Tốc độ (m/s)
                speed_kmh = speed_ms * 3.6  # Chuyển sang km/h
                return speed_kmh
        self.prev_positions[obj_id] = (center, current_frame_id)
        return None

    def process_video(self, video_path):
        """Xử lý video để phát hiện tốc độ."""
        cap = cv2.VideoCapture(video_path)
        frame_id = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame, conf=MODEL_CONFIG['confidence_threshold'])
            for det in results[0].boxes:
                if det.conf > MODEL_CONFIG['confidence_threshold']:
                    cls = int(det.cls)
                    label = self.model.names[cls]
                    if label in MODEL_CONFIG.get('vehicle_classes', {}).values():
                        bbox = det.xywh[0].tolist()  # [x_center, y_center, width, height]
                        speed = self.calculate_speed(bbox, frame_id)
                        if speed:
                            x, y, w, h = map(int, bbox)
                            cv2.rectangle(frame, (x - w//2, y - h//2), (x + w//2, y + h//2), (0, 255, 0), 2)
                            cv2.putText(frame, f"{label} {speed:.1f}km/h", (x - w//2, y - h//2 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            logger.info(f"Vehicle {label} speed: {speed:.1f} km/h")

            cv2.imshow('Speed Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame_id += 1

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = SpeedDetector()
    detector.process_video('sample_video.mp4')