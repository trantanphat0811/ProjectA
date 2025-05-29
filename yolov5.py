# yolov5.py - YOLOv5 integration for traffic detection
from ultralytics import YOLO
import platform

def load_yolov5_model(model_path='yolov5s.pt'):
    """Load a YOLOv5 model for traffic detection."""
    try:
        model = YOLO(model_path)
        print(f"Loaded YOLOv5 model: {model_path}")
        # Show device info for macOS
        if platform.system() == "Darwin":
            print("Running on macOS. For Apple Silicon, ensure 'mps' device is supported by your PyTorch backend.")
        return model
    except Exception as e:
        print(f"Failed to load YOLOv5 model: {e}")
        return None

if __name__ == "__main__":
    model = load_yolov5_model()
    if model:
        print("YOLOv5 model ready for traffic detection.")