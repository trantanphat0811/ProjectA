import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"
TEMP_DIR = PROJECT_ROOT / "temp"

for directory in [DATA_DIR, MODEL_DIR, OUTPUT_DIR, TEMP_DIR]:
    directory.mkdir(exist_ok=True)

MODEL_CONFIG = {
    'model_name': 'yolov8n.pt',
    'image_size': 640,
    'confidence_threshold': 0.5,
    'iou_threshold': 0.45,
    'max_detections': 100
}

TRAINING_CONFIG = {
    'epochs': 100,
    'batch_size': 16,
    'learning_rate': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'patience': 20,
    'save_period': 10
}

VEHICLE_CLASSES = {
    0: 'car',
    1: 'motorcycle', 
    2: 'bus',
    3: 'truck',
    4: 'bicycle'
}

CLASS_COLORS = {
    'car': (0, 255, 0),
    'motorcycle': (255, 0, 0),
    'bus': (0, 0, 255),
    'truck': (255, 255, 0),
    'bicycle': (255, 0, 255)
}

API_CONFIG = {
    'host': '0.0.0.0',
    'port': 8000,
    'title': 'Traffic Detection API',
    'version': '1.0.0',
    'description': 'API for vehicle detection and classification in traffic'
}

DASHBOARD_CONFIG = {
    'title': 'Traffic Vehicle Detection Dashboard',
    'page_icon': '🚗',
    'layout': 'wide',
    'sidebar_title': 'Controls'
}

VIDEO_CONFIG = {
    'fps': 30,
    'codec': 'mp4v',
    'max_frame_width': 1920,
    'max_frame_height': 1080,
    'webcam_index': 0
}

DATABASE_CONFIG = {
    'type': 'sqlite',
    'path': DATA_DIR / 'traffic_data.db',
    'tables': {
        'detections': 'detections',
        'statistics': 'statistics',
        'sessions': 'sessions'
    }
}

LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': PROJECT_ROOT / 'logs' / 'traffic_detection.log'
}

(PROJECT_ROOT / 'logs').mkdir(exist_ok=True)

DEBUG_MODE = os.getenv('DEBUG', 'False').lower() == 'true'
CUDA_AVAILABLE = os.getenv('CUDA_AVAILABLE', 'auto')
MAX_UPLOAD_SIZE = int(os.getenv('MAX_UPLOAD_SIZE', 100 * 1024 * 1024))
