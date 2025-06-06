import os
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Cấu hình cơ bản
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"
TEMP_DIR = PROJECT_ROOT / "temp"
LOGS_DIR = PROJECT_ROOT / "logs"

# Tạo các thư mục cơ bản với xử lý ngoại lệ
for directory in [DATA_DIR, MODELS_DIR, OUTPUT_DIR, TEMP_DIR, LOGS_DIR]:
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        print(f"Warning: Cannot create directory {directory}: {e}. Continuing with existing setup.")

# Cấu hình mô hình
MODEL_CONFIG = {
    'model_name': 'yolov8n.pt',
    'image_size': 640,
    'confidence_threshold': 0.5,
    'vehicle_classes': {0: 'car', 1: 'motorcycle', 2: 'bus', 3: 'truck', 4: 'bicycle'}  # Sửa key thành integer
}

# Cấu hình video
VIDEO_CONFIG = {
    'fps': 30,
    'scale_factor': 0.01,  # Hệ số tỷ lệ để tính tốc độ (cần hiệu chỉnh)
    'video_width': 1280,
    'video_height': 720,
    'roi': [0, int(0.6 * 720), 1280, 720]  # [x_min, y_min, x_max, y_max]
}

# Cập nhật ROI dựa trên chiều rộng và chiều cao video
def update_roi():
    """Cập nhật ROI dựa trên video_width và video_height."""
    width = VIDEO_CONFIG['video_width']
    height = VIDEO_CONFIG['video_height']
    if width <= 0 or height <= 0:
        print(f"Warning: Invalid video dimensions (width={width}, height={height}). Using default ROI.")
        return
    VIDEO_CONFIG['roi'] = [0, int(0.6 * height), width, height]

update_roi()  # Gọi hàm để cập nhật ROI ngay từ đầu

# Cấu hình tốc độ
SPEED_CONFIG = {
    'speed_limit': 60.0,  # Giới hạn tốc độ (km/h)
    'distance_per_pixel': 0.01  # Khoảng cách thực tế trên mỗi pixel (m/pixel, cần hiệu chỉnh)
}

# Cấu hình cơ sở dữ liệu
DATABASE_CONFIG = {
    'path': DATA_DIR / 'traffic_data.db',
    'table_name': 'detections'
}

# Cấu hình logging
LOGGING_CONFIG = {
    'level': logging.INFO,
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'handlers': [
        RotatingFileHandler(LOGS_DIR / 'traffic.log', maxBytes=10*1024*1024, backupCount=5),
        logging.StreamHandler()
    ]
}

# Khởi tạo logging sau khi đảm bảo thư mục LOGS_DIR tồn tại
logging.basicConfig(**LOGGING_CONFIG)

# Export cấu hình
__all__ = ['MODEL_CONFIG', 'VIDEO_CONFIG', 'SPEED_CONFIG', 'DATABASE_CONFIG', 'LOGGING_CONFIG']