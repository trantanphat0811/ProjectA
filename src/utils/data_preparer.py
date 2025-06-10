import os
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import yaml

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
    'model_name': 'yolov8n.pt',  # Tên file mô hình
    'confidence_threshold': 0.5,  # Ngưỡng độ tin cậy cho phát hiện
    'image_size': 640,  # Kích thước ảnh đầu vào
    'device': 'cuda',  # Device để chạy mô hình (cuda/cpu)
    'vehicle_classes': {  # Mapping class index với tên
        2: 'car',
        3: 'motorcycle',
        5: 'bus',
        7: 'truck'
    }
}

# Cấu hình video
VIDEO_CONFIG = {
    'input_size': (1920, 1080),  # Kích thước đầu vào
    'output_size': (1280, 720),  # Kích thước đầu ra
    'fps': 30,  # Frame rate
    'codec': 'mp4v',  # Codec video
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
    'path': 'detections.db'
}

# Cấu hình logging
LOGGING_CONFIG = {
    'filename': 'logs/speed_detection.log',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'level': logging.INFO,
    'filemode': 'a'
}

# Khởi tạo logging sau khi đảm bảo thư mục LOGS_DIR tồn tại
logging.basicConfig(**LOGGING_CONFIG)

# Export cấu hình
__all__ = ['MODEL_CONFIG', 'VIDEO_CONFIG', 'SPEED_CONFIG', 'DATABASE_CONFIG', 'LOGGING_CONFIG']

def prepare_dataset(data_dir: str, split_ratio: float = 0.8):
    """Chuẩn bị dataset cho training."""
    data_dir = Path(data_dir)
    
    # Tạo cấu trúc thư mục
    (data_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (data_dir / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (data_dir / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (data_dir / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
    
    # Tạo file cấu hình dataset
    data_yaml = {
        'path': str(data_dir),
        'train': 'images/train',
        'val': 'images/val',
        'names': {
            0: 'person',
            1: 'bicycle',
            2: 'car',
            3: 'motorcycle',
            4: 'airplane',
            5: 'bus',
            6: 'train',
            7: 'truck',
            8: 'boat'
        }
    }
    
    with open(data_dir / 'data.yaml', 'w') as f:
        yaml.dump(data_yaml, f, sort_keys=False)
    
    return str(data_dir / 'data.yaml')

def create_model_config():
    """Tạo file cấu hình cho model."""
    config = {
        'nc': len(MODEL_CONFIG['vehicle_classes']),  # Số lượng class
        'depth_multiple': 0.33,  # Hệ số độ sâu của model
        'width_multiple': 0.50,  # Hệ số độ rộng của model
        'anchors': 3,  # Số lượng anchor box
        'backbone': [
            # [from, number, module, args]
            [-1, 1, 'Conv', [64, 6, 2, 2]],  # 0-P1/2
            [-1, 3, 'Conv', [128, 3, 2]],     # 1-P2/4
            [-1, 6, 'Conv', [256, 3, 2]],     # 2-P3/8
            [-1, 9, 'Conv', [512, 3, 2]],     # 3-P4/16
            [-1, 3, 'Conv', [1024, 3, 2]],    # 4-P5/32
        ],
        'head': [
            [-1, 3, 'Conv', [1024, 3, 1]],    # 5
            [-1, 1, 'Conv', [len(MODEL_CONFIG['vehicle_classes']) * 3, 1, 1]],  # 6
        ]
    }
    
    config_path = 'models/yolov8_vehicle.yaml'
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    
    return config_path

def prepare_training_config():
    """Tạo file cấu hình cho training."""
    config = {
        'epochs': 100,
        'batch_size': 16,
        'imgsz': MODEL_CONFIG['image_size'],
        'optimizer': 'SGD',
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 0.05,
        'cls': 0.5,
        'cls_pw': 1.0,
        'obj': 1.0,
        'obj_pw': 1.0,
        'iou_t': 0.20,
        'anchor_t': 4.0,
        'fl_gamma': 0.0,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,
        'copy_paste': 0.0
    }
    
    config_path = 'models/training_config.yaml'
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    
    return config_path

if __name__ == "__main__":
    # Tạo cấu trúc thư mục và file cấu hình
    data_yaml = prepare_dataset('data')
    model_yaml = create_model_config()
    train_yaml = prepare_training_config()
    
    print(f"Created dataset config: {data_yaml}")
    print(f"Created model config: {model_yaml}")
    print(f"Created training config: {train_yaml}")