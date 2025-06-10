import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
TEMP_FOLDER = os.path.join(BASE_DIR, 'temp')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
VIOLATION_IMAGE_DIR = os.path.join(BASE_DIR, 'static', 'violations')

# Create necessary directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(VIOLATION_IMAGE_DIR, exist_ok=True)

# Database configuration
DATABASE_URL = 'sqlite:///detections.db'

# Logging configuration
LOG_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',
        },
        'file': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': os.path.join(BASE_DIR, 'logs', 'app.log'),
            'mode': 'a',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default', 'file'],
            'level': 'INFO',
            'propagate': True
        }
    }
}

# Model configuration
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'yolov8n.pt')
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.7

# Speed detection configuration
SPEED_LIMIT = 40  # km/h
FRAME_RATE = 30  # fps
PIXEL_TO_METER_RATIO = 0.1  # meters per pixel

# Video settings
DETECTION_CONFIDENCE = 0.5
TRACKING_CONFIDENCE = 0.5
MAX_DISAPPEARED = 30
MIN_DETECTION_CONFIDENCE = 0.6

# Speed calculation settings
SPEED_CALCULATION = {
    'distance_meters': 20,  # Distance between detection lines in meters
    'fps': 30,  # Video frame rate
    'speed_limit': 50,  # Speed limit in km/h
}

# Violation settings
VIOLATION_THRESHOLD = 1.1  # 10% above speed limit

# File paths
TEMP_FOLDER = os.path.join(BASE_DIR, 'temp')
LOG_FOLDER = os.path.join(BASE_DIR, 'logs')
MODEL_PATH = os.path.join(BASE_DIR, 'models/yolov8n.pt')

# Ensure directories exist
for folder in [UPLOAD_FOLDER, TEMP_FOLDER, LOG_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Camera configuration
CAMERA_CONFIG = {
    'resolution': (1920, 1080),
    'fps': 30,
}

# Storage configuration
STORAGE_CONFIG = {
    'max_video_size': 500 * 1024 * 1024,  # 500MB
    'allowed_extensions': {'mp4', 'avi', 'mov'},
}

# Tracking configuration
TRACKING_CONFIG = {
    'confidence_threshold': 0.5,
    'iou_threshold': 0.3,
    'max_age': 70,
    'min_hits': 3,
    'detection_classes': [2, 3, 5, 7],  # car, motorcycle, bus, truck
}

# Speed limits for different vehicle types (in km/h)
SPEED_LIMITS = {
    'car': 50,
    'motorcycle': 50,
    'bus': 40,
    'truck': 40,
} 