import cv2
import numpy as np
from ultralytics import YOLO
import onnxruntime as ort
import logging
from data_preparer import MODEL_CONFIG, VEHICLE_CLASSES, VIDEO_CONFIG, LOGGING_CONFIG  # Sửa import

# Cấu hình logging
logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)

class RealtimeTrafficProcessor:
    def __init__(self, model_path=None, webcam_index=0, use_onnx=False):
        """Khởi tạo bộ xử lý video thời gian thực."""
        self.model_path = model_path or MODEL_CONFIG['model_name']
        self.webcam_index = webcam_index or VIDEO_CONFIG['webcam_index']
        self.use_onnx = use_onnx
        self.onnx_session = None

        if self.use_onnx and self.model_path.endswith('.onnx'):
            try:
                self.onnx_session = ort.InferenceSession(self.model_path)
                logger.info(f"Đã tải mô hình ONNX: {self.model_path}")
            except Exception as e:
                logger.error(f"Không thể tải mô hình ONNX: {e}")
                raise Exception(f"Không thể tải mô hình ONNX: {e}")
        else:
            try:
                self.model = YOLO(self.model_path)
                logger.info(f"Đã tải mô hình YOLO: {self.model_path}")
            except Exception as e:
                logger.error(f"Không thể tải mô hình YOLO: {e}")
                raise Exception(f"Không thể tải mô hình YOLO: {e}")
        
        self.cap = cv2.VideoCapture(self.webcam_index)
        if not self.cap.isOpened():
            logger.error(f"Không thể mở webcam index {self.webcam_index}")
            raise Exception(f"Không thể mở webcam index {self.webcam_index}")

    def process_realtime(self):
        """Xử lý luồng video từ webcam và hiển thị kết quả."""
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Không thể đọc frame từ webcam")
                    break
                
                # Resize frame
                frame = cv2.resize(frame, (MODEL_CONFIG['image_size'], MODEL_CONFIG['image_size']))
                
                if self.use_onnx and self.onnx_session:
                    # ONNX inference
                    img = frame.transpose(2, 0, 1).astype(np.float32) / 255.0
                    img = np.expand_dims(img, axis=0)
                    input_name = self.onnx_session.get_inputs()[0].name
                    outputs = self.onnx_session.run(None, {input_name: img})[0]
                    # Giả định outputs có định dạng [boxes, scores, classes]
                    boxes, scores, classes = outputs[:4], outputs[4], outputs[5]
                    annotated_frame = self.draw_boxes(frame, boxes, scores, classes)
                else:
                    # YOLO inference
                    results = self.model(frame, conf=MODEL_CONFIG['confidence_threshold'], iou=MODEL_CONFIG['iou_threshold'])
                    annotated_frame = results[0].plot()

                cv2.imshow('Realtime Traffic Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            logger.error(f"Lỗi khi xử lý video thời gian thực: {e}")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            logger.info("Đã dừng xử lý video thời gian thực")

    def draw_boxes(self, frame, boxes, scores, classes):
        """Vẽ bounding box cho ONNX output."""
        for box, score, cls in zip(boxes, scores, classes):
            if score > MODEL_CONFIG['confidence_threshold']:
                x1, y1, x2, y2 = map(int, box)
                label = VEHICLE_CLASSES.get(int(cls), 'unknown')
                color = (0, 255, 0)  # Màu mặc định
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame

if __name__ == "__main__":
    # Sử dụng ONNX nếu có file .onnx
    processor = RealtimeTrafficProcessor(model_path='path/to/yolov8n.onnx', use_onnx=True)
    processor.process_realtime()