import numpy as np
from ultralytics import YOLO
from pathlib import Path
import pandas as pd
import logging
from datetime import datetime
from itertools import product
import random

# Import configurations from your project
from config import MODEL_CONFIG, TRAINING_CONFIG, LOGGING_CONFIG

class TrafficDetectionTrainer:
    def __init__(self, model_name=None, log_dir='logs', config=None):
        """
        Initialize the Traffic Detection Trainer.
        
        Args:
            model_name (str): Name of the YOLO model (e.g., 'yolov8n.pt').
            log_dir (str): Directory to store log files.
            config (dict): Optional configuration dictionary for training parameters.
        """
        self.model_name = model_name or MODEL_CONFIG.get('model_name', 'yolov8n.pt')
        # Device selection for macOS
        try:
            import torch
            if torch.backends.mps.is_available():
                self.device = 'mps'  # Use MPS for Apple Silicon
            elif torch.cuda.is_available():
                self.device = 'cuda'  # Fallback to CUDA (unlikely on macOS)
            else:
                self.device = 'cpu'  # Default to CPU
        except ImportError:
            self.device = 'cpu'  # Fallback if torch is not installed
        self.model = None
        self.config = config or TRAINING_CONFIG

        # Setup logging
        Path(log_dir).mkdir(exist_ok=True)
        log_file = Path(log_dir) / f'tuning_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        logging.basicConfig(
            filename=str(log_file),
            level=LOGGING_CONFIG.get('level', logging.INFO),
            format=LOGGING_CONFIG.get('format', '%(asctime)s - %(levelname)s - %(message)s')
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized trainer with model: {self.model_name}, device: {self.device}")
        print(f"Using device: {self.device}")
        print(f"Model: {self.model_name}")

    def fine_tune_hyperparameters(self, config_path, trials=10, save_results=True, results_file='tuning_results.csv'):
        """
        Automated hyperparameter tuning for YOLO model.
        
        Args:
            config_path (str): Path to the dataset configuration file (data.yaml).
            trials (int): Number of trials for hyperparameter tuning.
            save_results (bool): Whether to save tuning results to a CSV file.
            results_file (str): File path to save tuning results.
        
        Returns:
            tuple: (best_params, results) - Best parameters and list of all trial results.
        """
        # Validate inputs
        config_path = Path(config_path)
        if not config_path.exists():
            self.logger.error(f"Config file not found: {config_path}")
            raise FileNotFoundError(f"Config file not found: {config_path}")
        if trials < 1:
            self.logger.error("Number of trials must be positive")
            raise ValueError("Number of trials must be positive")

        print(f"Starting hyperparameter tuning ({trials} trials)...")
        self.logger.info(f"Starting hyperparameter tuning with {trials} trials")

        # Define parameter ranges
        param_ranges = {
            'lr0': self.config.get('learning_rate_range', [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]),
            'momentum': self.config.get('momentum_range', [0.8, 0.85, 0.9, 0.95]),
            'weight_decay': self.config.get('weight_decay_range', [0.0, 0.0001, 0.0005, 0.001])
        }

        # Default training parameters
        default_params = {
            'epochs': self.config.get('epochs', 20),
            'batch': self.config.get('batch_size', 16),
            'imgsz': MODEL_CONFIG.get('image_size', 640)
        }

        best_score = 0
        best_params = None
        results = []

        # Generate unique parameter combinations
        param_combinations = list(product(*[param_ranges[param] for param in param_ranges]))
        random.shuffle(param_combinations)
        trials = min(trials, len(param_combinations))  # Limit trials to available combinations

        for trial, param_combo in enumerate(param_combinations[:trials], 1):
            trial_params = {
                'lr0': param_combo[0],
                'momentum': param_combo[1],
                'weight_decay': param_combo[2]
            }
            print(f"\nTrial {trial}/{trials}: {trial_params}")
            self.logger.info(f"Trial {trial}/{trials}: {trial_params}")

            try:
                # Initialize new model for each trial
                model = YOLO(self.model_name)
                
                # Train with current parameters
                result = model.train(
                    data=str(config_path),
                    epochs=default_params['epochs'],
                    batch=default_params['batch'],
                    imgsz=default_params['imgsz'],
                    device=self.device,
                    verbose=False,
                    project='runs/tune',
                    name=f'trial_{trial}_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{trial:03d}',
                    **trial_params
                )

                # Evaluate model
                val_result = model.val(data=str(config_path), device=self.device, verbose=False)
                score = val_result.box.map50 if hasattr(val_result, 'box') else 0
                map50_95 = val_result.box.map if hasattr(val_result, 'box') else 0

                results.append({
                    'trial': trial,
                    'params': trial_params,
                    'map50': score,
                    'map50_95': map50_95
                })

                if score > best_score:
                    best_score = score
                    best_params = trial_params
                    print(f"New best score (mAP50): {score:.4f}")
                    self.logger.info(f"New best score (mAP50): {score:.4f} with params: {trial_params}")

            except Exception as e:
                print(f"Trial {trial} failed: {e}")
                self.logger.error(f"Trial {trial} failed: {e}")
                continue

        # Print best results
        if best_params:
            print(f"\nBest parameters: {best_params}")
            print(f"Best mAP50 score: {best_score:.4f}")
            self.logger.info(f"Best parameters: {best_params}, Best mAP50 score: {best_score:.4f}")
        else:
            print("No successful trials completed")
            self.logger.warning("No successful trials completed")

        # Save results to CSV if requested
        if save_results and results:
            try:
                df = pd.DataFrame(results)
                df.to_csv(results_file, index=False)
                print(f"Tuning results saved to {results_file}")
                self.logger.info(f"Tuning results saved to {results_file}")
            except Exception as e:
                self.logger.error(f"Failed to save results: {e}")
                print(f"Failed to save results: {e}")

        return best_params, results