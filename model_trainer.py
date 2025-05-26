import torch
import torch.nn as nn
from ultralytics import YOLO
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import logging

from config import MODEL_DIR, TRAINING_CONFIG, MODEL_CONFIG, LOGGING_CONFIG


class TrafficDetectionTrainer:
    def __init__(self, model_name=None):
        self.model_name = model_name or MODEL_CONFIG['model_name']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.training_results = None
        logging.basicConfig(**LOGGING_CONFIG)
        self.logger = logging.getLogger(__name__)
        print(f"Using device: {self.device}")
        print(f"Model: {self.model_name}")
    
    def load_model(self, model_path=None):
        try:
            if model_path and Path(model_path).exists():
                self.model = YOLO(str(model_path))
                print(f"Loaded custom model: {model_path}")
            else:
                self.model = YOLO(self.model_name)
                print(f"Loaded pretrained model: {self.model_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
    
    def train_model(self, config_path, **kwargs):
        if not self.model:
            self.load_model()
        train_params = {**TRAINING_CONFIG, **kwargs}
        print("Starting model training...")
        print(f"Training parameters: {train_params}")
        try:
            self.training_results = self.model.train(
                data=str(config_path),
                epochs=train_params['epochs'],
                batch=train_params['batch_size'],
                imgsz=MODEL_CONFIG['image_size'],
                device=self.device,
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
        if not self.model:
            print("No model loaded for validation")
            return None
        print("Validating model...")
        try:
            metrics = self.model.val(
                data=str(config_path),
                device=self.device,
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
            print("No model loaded for export")
            return None
        print(f"Exporting model to {format} format...")
        try:
            export_path = self.model.export(
                format=format,
                device=self.device,
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
        try:
            self.model = YOLO(str(checkpoint_path))
            print(f"Resuming training from: {checkpoint_path}")
            results = self.model.train(
                resume=True,
                **kwargs
            )
            print("Training resumed successfully!")
            return results
        except Exception as e:
            print(f"Failed to resume training: {e}")
            return None
    
    def benchmark_model(self, test_images_dir):
        if not self.model:
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
                model = YOLO(self.model_name)
                result = model.train(
                    data=str(config_path),
                    epochs=20,
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
        trainer.export_model('engine')
        trainer.plot_training_results()
        print("Training pipeline completed successfully!")
    else:
        print("Training pipeline failed!")


if __name__ == "__main__":
    main()