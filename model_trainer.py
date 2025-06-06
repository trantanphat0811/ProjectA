import numpy as np
from ultralytics import YOLO
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from datetime import datetime
import logging
import onnxruntime as ort
import cv2

from config import MODEL_DIR, TRAINING_CONFIG, MODEL_CONFIG, LOGGING_CONFIG

class TrafficDetectionTrainer:
    def __init__(self, model_name=None):
        self.model_name = model_name or MODEL_CONFIG['model_name']
        self.device = 'cpu'  # ONNX Runtime uses CPU on macOS; MPS/CUDA optional
        self.model = None
        self.onnx_session = None
        self.training_results = None
        logging.basicConfig(**LOGGING_CONFIG)
        self.logger = logging.getLogger(__name__)
        print(f"Using device: {self.device}")
        print(f"Model: {self.model_name}")

    def load_model(self, model_path=None):
        try:
            if model_path and Path(model_path).exists() and model_path.endswith('.onnx'):
                # Load ONNX model for inference
                self.onnx_session = ort.InferenceSession(str(model_path))
                print(f"Loaded ONNX model: {model_path}")
            else:
                # Load YOLO model (PyTorch) for training or export
                self.model = YOLO(self.model_name)
                print(f"Loaded pretrained YOLO model: {self.model_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False

    def train_model(self, config_path, **kwargs):
        if not self.model:
            self.load_model()
        if self.onnx_session:
            print("ONNX model cannot be used for training")
            return None
        train_params = {**TRAINING_CONFIG, **kwargs}
        print("Starting model training...")
        print(f"Training parameters: {train_params}")
        try:
            import torch
            self.training_results = self.model.train(
                data=str(config_path),
                epochs=train_params['epochs'],
                batch=train_params['batch_size'],
                imgsz=MODEL_CONFIG['image_size'],
                device='mps' if torch.backends.mps.is_available() else 'cpu',
                project=str(MODEL_DIR),
                name=f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                save_period=train_params['save_period'],
                patience=train_params['patience'],
                lr0=train_params['learning_rate'],
                momentum=train_params['momentum'],
                weight_decay=train_params['weight_decay'],
                verbose=True,
                plots=True
            )
            print("Training completed successfully!")
            self.logger.info("Model training completed")
            return self.training_results
        except Exception as e:
            print(f"Training failed: {e}")
            self.logger.error(f"Training failed: {e}")
            return None

    def validate_model(self, config_path):
        if self.onnx_session:
            print("ONNX validation not implemented; use PyTorch model for validation")
            return None
        if not self.model:
            print("No PyTorch model loaded for validation")
            return None
        print("Validating model...")
        try:
            import torch
            metrics = self.model.val(
                data=str(config_path),
                device='mps' if torch.backends.mps.is_available() else 'cpu',
                plots=True,
                save_json=True
            )
            print("Validation completed!")
            self._print_validation_metrics(metrics)
            return metrics
        except Exception as e:
            print(f"Validation failed: {e}")
            self.logger.error(f"Validation failed: {e}")
            return None

    def _print_validation_metrics(self, metrics):
        print("\nValidation Metrics:")
        print("=" * 40)
        try:
            if hasattr(metrics, 'box'):
                box_metrics = metrics.box
                print(f"mAP50: {box_metrics.map50:.4f}")
                print(f"mAP50-95: {box_metrics.map:.4f}")
                print(f"Precision: {box_metrics.mp:.4f}")
                print(f"Recall: {box_metrics.mr:.4f}")
            else:
                print("Detailed metrics not available")
        except Exception as e:
            print(f"Error displaying metrics: {e}")

    def export_model(self, format='onnx', **kwargs):
        if not self.model:
            print("No PyTorch model loaded for export")
            return None
        print(f"Exporting model to {format} format...")
        try:
            import torch
            export_path = self.model.export(
                format=format,
                device='mps' if torch.backends.mps.is_available() else 'cpu',
                **kwargs
            )
            print(f"Model exported successfully: {export_path}")
            self.logger.info(f"Model exported to {format}: {export_path}")
            return export_path
        except Exception as e:
            print(f"Export failed: {e}")
            self.logger.error(f"Export failed: {e}")
            return None

    def plot_training_results(self, save_path=None):
        if not self.training_results:
            print("No training results to plot")
            return
        try:
            print("Training plots saved automatically by YOLO")
            if save_path:
                self._create_training_summary_plot(save_path)
        except Exception as e:
            print(f"Failed to plot results: {e}")

    def _create_training_summary_plot(self, save_path):
        pass

    def resume_training(self, checkpoint_path, **kwargs):
        if self.onnx_session:
            print("ONNX model cannot be used for training")
            return None
        try:
            import torch
            self.model = YOLO(str(checkpoint_path))
            print(f"Resuming training from: {checkpoint_path}")
            results = self.model.train(
                resume=True,
                device='mps' if torch.backends.mps.is_available() else 'cpu',
                **kwargs
            )
            print("Training resumed successfully!")
            return results
        except Exception as e:
            print(f"Failed to resume training: {e}")
            return None

    def benchmark_model(self, test_images_dir):
        if not self.onnx_session and not self.model:
            print("No model loaded for benchmarking")
            return None
        print("Benchmarking model performance...")
        try:
            import time
            test_images = list(Path(test_images_dir).glob('*.jpg'))
            if not test_images:
                print("No test images found")
                return None
            times = []
            if self.onnx_session:
                # ONNX Runtime inference
                input_name = self.onnx_session.get_inputs()[0].name
                for img_path in test_images[:10]:
                    img = cv2.imread(str(img_path))
                    img = cv2.resize(img, (MODEL_CONFIG['image_size'], MODEL_CONFIG['image_size']))
                    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
                    img = np.expand_dims(img, axis=0)
                    start_time = time.time()
                    outputs = self.onnx_session.run(None, {input_name: img})
                    end_time = time.time()
                    times.append(end_time - start_time)
            else:
                # PyTorch inference
                for img_path in test_images[:10]:
                    start_time = time.time()
                    results = self.model(str(img_path))
                    end_time = time.time()
                    times.append(end_time - start_time)
            avg_time = np.mean(times)
            fps = 1.0 / avg_time
            print(f"Average inference time: {avg_time:.4f}s")
            print(f"Average FPS: {fps:.2f}")
            return {'avg_time': avg_time, 'fps': fps, 'times': times}
        except Exception as e:
            print(f"Benchmarking failed: {e}")
            return None

    def fine_tune_hyperparameters(self, config_path, trials=10):
        if self.onnx_session:
            print("ONNX model cannot be used for hyperparameter tuning")
            return None, []
        print(f"Starting hyperparameter tuning ({trials} trials)...")
        param_ranges = {
            'lr0': [0.001, 0.01, 0.1],
            'momentum': [0.8, 0.9, 0.95],
            'weight_decay': [0.0001, 0.0005, 0.001]
        }
        best_score = 0
        best_params = None
        results = []
        for trial in range(trials):
            trial_params = {}
            for param, values in param_ranges.items():
                trial_params[param] = np.random.choice(values)
            print(f"\nTrial {trial + 1}/{trials}: {trial_params}")
            try:
                import torch
                model = YOLO(self.model_name)
                result = model.train(
                    data=str(config_path),
                    epochs=20,
                    device='mps' if torch.backends.mps.is_available() else 'cpu',
                    **trial_params,
                    verbose=False
                )
                val_result = model.val(data=str(config_path), verbose=False)
                score = val_result.box.map50 if hasattr(val_result, 'box') else 0
                results.append({
                    'trial': trial + 1,
                    'params': trial_params,
                    'score': score
                })
                if score > best_score:
                    best_score = score
                    best_params = trial_params
                    print(f"New best score: {score:.4f}")
            except Exception as e:
                print(f"Trial {trial + 1} failed: {e}")
                continue
        print(f"\nBest parameters: {best_params}")
        print(f"Best score: {best_score:.4f}")
        return best_params, results

def main():
    print("Starting Traffic Detection Model Training")
    print("=" * 50)
    trainer = TrafficDetectionTrainer()
    config_path = Path("data/dataset.yaml")
    if not config_path.exists():
        print("Dataset config not found. Run data_preparer.py first!")
        return
    if not trainer.load_model():
        return
    results = trainer.train_model(config_path, epochs=50)
    if results:
        metrics = trainer.validate_model(config_path)
        trainer.export_model('onnx')
        # Load ONNX model for inference
        trainer.load_model(MODEL_DIR / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}/weights/best.onnx")
        trainer.benchmark_model("data/images/val")
        print("Training pipeline completed successfully!")
    else:
        print("Training pipeline failed!")

if __name__ == "__main__":
    main()