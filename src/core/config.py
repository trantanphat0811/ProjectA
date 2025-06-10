import os
from pathlib import Path

# Project root path
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Speed calculation configuration
SPEED_CALCULATION = {
    'distance_reference': 20,  # Reference distance (meters)
    'min_frames': 10,  # Minimum frames for speed calculation
    'max_frames': 30,  # Maximum frames for speed calculation
    'smoothing_factor': 0.3,  # Speed smoothing factor
    'confidence_threshold': 0.7  # Confidence threshold
}

# Speed calculation variables
DISTANCE_REFERENCE = SPEED_CALCULATION['distance_reference']
MIN_FRAMES_FOR_SPEED = SPEED_CALCULATION['min_frames']
MAX_FRAMES_FOR_SPEED = SPEED_CALCULATION['max_frames']
SPEED_SMOOTHING_FACTOR = SPEED_CALCULATION['smoothing_factor']
CONFIDENCE_THRESHOLD = SPEED_CALCULATION['confidence_threshold']

# Camera and video configuration
CAMERA_CONFIG = {
    'fps': 30,
    'width': 1920,
    'height': 1080,
    'exposure': 0,
    'contrast': 1.0,
    'roi': {
        'x1': 100,
        'y1': 200,
        'x2': 1820,
        'y2': 880
    }
}

# Tracking configuration
TRACKING_CONFIG = {
    'tracker': 'bytetrack.yaml',
    'conf_thres': 0.5,
    'iou_thres': 0.4,
    'max_age': 30,
    'min_hits': 3,
    'filter_classes': ['car', 'truck', 'bus', 'motorcycle']
}

# Speed limits for each vehicle type
SPEED_LIMITS = {
    'car': 60,
    'truck': 50,
    'bus': 50,
    'motorcycle': 40
}

# Storage configuration
STORAGE_CONFIG = {
    'upload_dir': PROJECT_ROOT / 'uploads',
    'processed_dir': PROJECT_ROOT / 'static' / 'processed',
    'violations_dir': PROJECT_ROOT / 'static' / 'violations',
    'database_path': PROJECT_ROOT / 'detections.db',
    'model_path': PROJECT_ROOT / 'models' / 'yolov8n.pt',
    'temp_dir': PROJECT_ROOT / 'temp',
    'max_storage_days': 30,
    'backup_interval': 24
}

# Statistics configuration
STATISTICS_CONFIG = {
    'update_interval': 1,  # Update statistics every second
    'speed_window': 30,    # Time window for average speed calculation (frames)
    'clear_inactive_after': 60  # Clear inactive objects after 60 frames
}

# Violations directory
VIOLATIONS_DIR = PROJECT_ROOT / 'static' / 'violations'

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'filename': PROJECT_ROOT / 'logs' / 'speed_detection.log',
    'filemode': 'a'
}

# Display configuration
DISPLAY_CONFIG = {
    'show_tracking': True,  # Show tracking lines
    'show_speed': True,  # Show speed
    'show_violations': True,  # Show violation warnings
    'text_scale': 0.6,  # Text size
    'line_thickness': 2,  # Line thickness
    'colors': {
        'normal': (0, 255, 0),  # Color for normal objects
        'violation': (0, 0, 255),  # Color for violating objects
        'roi': (255, 255, 0),  # Color for ROI
        'trajectory': (255, 0, 0)  # Color for trajectory
    }
}

# Model configuration
MODEL_CONFIG = {
    'model_type': 'yolov8',
    'model_size': 'n',  # nano
    'input_size': (640, 640),
    'device': 'cuda:0' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu',
    'half_precision': True,
    'classes': TRACKING_CONFIG['filter_classes']
}

# Training configuration
TRAINING_CONFIG = {
    'epochs': 100,
    'batch_size': 16,
    'learning_rate': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3,
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1
}

# Create necessary directories
for directory in [
    STORAGE_CONFIG['upload_dir'],
    STORAGE_CONFIG['processed_dir'],
    STORAGE_CONFIG['violations_dir'],
    STORAGE_CONFIG['temp_dir'],
    PROJECT_ROOT / 'logs'
]:
    directory.mkdir(parents=True, exist_ok=True)