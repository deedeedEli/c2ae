import yaml
import os
from typing import Any, Dict, Optional
from pathlib import Path


class Config:
    """Configuration manager for C2AE experiments."""
    
    def __init__(self, config_path: Optional[str] = None, overrides: Optional[Dict] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to YAML config file
            overrides: Dictionary of config overrides
        """
        self.config = self._load_default_config()
        
        if config_path:
            self.load_from_file(config_path)
        
        if overrides:
            self.update(overrides)
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration."""
        return {
            'model': {
                'encoder_arch': 'resnet18',
                'embedding_dim': 128,
                'latent_dim': 512,
                'film_positions': [0, 1, 2, 3]
            },
            'training': {
                'num_epochs': 100,
                'batch_size': 128,
                'learning_rate': 0.001,
                'weight_decay': 0.0001,
                'optimizer': 'adam',
                'scheduler': 'step',
                'scheduler_params': {
                    'step_size': 30,
                    'gamma': 0.1
                },
                'reconstruction_weight': 1.0,
                'early_stopping_patience': 20,
                'early_stopping_metric': 'val_loss'
            },
            'data': {
                'dataset': 'cifar10',
                'data_root': './data',
                'num_workers': 4,
                'pin_memory': True,
                'num_known_classes': 6,
                'known_class_indices': None,
                'use_background': True,
                'background_ratio': 0.3,
                'background_augmentation': 0.8
            },
            'augmentation': {
                'train': {
                    'random_crop': True,
                    'crop_padding': 4,
                    'random_horizontal_flip': True,
                    'flip_probability': 0.5,
                    'color_jitter': True,
                    'jitter_params': {
                        'brightness': 0.2,
                        'contrast': 0.2,
                        'saturation': 0.2,
                        'hue': 0.1
                    },
                    'normalize': True,
                    'normalization_mean': [0.5, 0.5, 0.5],
                    'normalization_std': [0.5, 0.5, 0.5]
                },
                'test': {
                    'normalize': True,
                    'normalization_mean': [0.5, 0.5, 0.5],
                    'normalization_std': [0.5, 0.5, 0.5]
                }
            },
            'evaluation': {
                'metrics': ['auroc', 'aupr', 'ccr_at_fpr', 'f1_score'],
                'fpr_thresholds': [0.01, 0.05, 0.1],
                'threshold_selection': 'optimal_f1',
                'score_aggregation': 'min'
            },
            'experiment': {
                'name': 'c2ae_default',
                'seed': 42,
                'log_interval': 10,
                'checkpoint_interval': 10,
                'save_best_only': True,
                'use_tensorboard': True,
                'tensorboard_dir': './results/tensorboard',
                'save_dir': './results'
            },
            'hardware': {
                'device': 'cuda',
                'cuda_deterministic': False,
                'num_gpus': 1
            }
        }
    
    def load_from_file(self, config_path: str):
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            file_config = yaml.safe_load(f)
        self._deep_update(self.config, file_config)
    
    def _deep_update(self, base: Dict, update: Dict):
        """Recursively update nested dictionary."""
        for key, value in update.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value
    
    def update(self, overrides: Dict):
        """Update configuration with overrides."""
        self._deep_update(self.config, overrides)
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Example: config.get('model.encoder_arch') -> 'resnet18'
        """
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def save(self, save_path: str):
        """Save current configuration to file."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self.config[key]
    
    def __repr__(self) -> str:
        return f"Config({yaml.dump(self.config, default_flow_style=False)})"
