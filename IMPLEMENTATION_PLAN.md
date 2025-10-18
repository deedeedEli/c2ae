# IMPLEMENTATION_PLAN.md - Step-by-Step Execution Guide

## Implementation Overview

This guide provides a sequential, phase-based roadmap for implementing the C2AE (Class Conditioned Auto-Encoder) paper reproduction. The implementation is structured in 4 major phases with clear dependencies and validation checkpoints. Each phase builds upon the previous, ensuring a solid foundation before advancing.

**Total Estimated Implementation Time**: 16-20 hours
**Recommended Order**: Follow phases sequentially, completing all validation checkpoints before proceeding.

---

## Phase 1: Core Infrastructure (2-3 hours)

### 1.1 Project Initialization

**CLAUDE CODE TASK**: Create the complete project directory structure as defined in PROJECT_STRUCTURE.md.

```python
# File: setup.py
from setuptools import setup, find_packages

setup(
    name="c2ae-reproduction",
    version="1.0.0",
    description="Reproduction of C2AE: Class Conditioned Auto-Encoder for Open-Set Recognition",
    author="Reproduction Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.2.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tensorboard>=2.12.0",
        "PyYAML>=6.0",
        "tqdm>=4.65.0",
        "Pillow>=9.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.0.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.2.0",
            "jupyter>=1.0.0",
        ]
    }
)
```

```bash
# File: scripts/setup_environment.sh
#!/bin/bash
set -e

echo "Setting up C2AE reproduction environment..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -e ".[dev]"

# Create necessary directories
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/openset_splits
mkdir -p results/checkpoints
mkdir -p results/logs
mkdir -p results/metrics
mkdir -p results/visualizations
mkdir -p results/tensorboard

echo "Environment setup complete!"
echo "Activate with: source venv/bin/activate"
```

### 1.2 Configuration System

**CLAUDE CODE TASK**: Implement configuration management system.

```python
# File: src/utils/config.py
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
```

### 1.3 Reproducibility Utilities

**CLAUDE CODE TASK**: Implement reproducibility and device management utilities.

```python
# File: src/utils/reproducibility.py
import random
import numpy as np
import torch
import os

def set_seed(seed: int, deterministic: bool = False):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
        deterministic: If True, use deterministic CUDA operations (slower but reproducible)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    else:
        torch.backends.cudnn.benchmark = True

def get_reproducibility_info():
    """Get information about current reproducibility settings."""
    info = {
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
    }
    return info
```

```python
# File: src/utils/device_manager.py
import torch
from typing import Union, List

class DeviceManager:
    """Manages device allocation for training and inference."""
    
    def __init__(self, device: str = 'cuda', num_gpus: int = 1):
        """
        Initialize device manager.
        
        Args:
            device: 'cuda', 'cpu', or 'mps' (for Apple Silicon)
            num_gpus: Number of GPUs to use (for multi-GPU training)
        """
        self.device_type = device
        self.num_gpus = num_gpus
        self.device = self._get_device()
    
    def _get_device(self) -> torch.device:
        """Get appropriate device."""
        if self.device_type == 'cuda':
            if torch.cuda.is_available():
                return torch.device('cuda')
            else:
                print("CUDA requested but not available, falling back to CPU")
                return torch.device('cpu')
        elif self.device_type == 'mps':
            if torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                print("MPS requested but not available, falling back to CPU")
                return torch.device('cpu')
        else:
            return torch.device('cpu')
    
    def to_device(self, tensor_or_model: Union[torch.Tensor, torch.nn.Module]):
        """Move tensor or model to device."""
        return tensor_or_model.to(self.device)
    
    def get_device_info(self) -> dict:
        """Get information about current device."""
        info = {
            'device_type': str(self.device),
            'cuda_available': torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            info.update({
                'cuda_device_count': torch.cuda.device_count(),
                'cuda_device_name': torch.cuda.get_device_name(0),
                'cuda_memory_allocated': torch.cuda.memory_allocated(0),
                'cuda_memory_cached': torch.cuda.memory_reserved(0),
            })
        
        return info
```

### 1.4 Checkpoint Management

**CLAUDE CODE TASK**: Implement checkpoint saving and loading.

```python
# File: src/utils/checkpoint.py
import torch
import os
from pathlib import Path
from typing import Dict, Optional, Any

class CheckpointManager:
    """Manages model checkpoint saving and loading."""
    
    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 5):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.checkpoint_history = []
    
    def save_checkpoint(self, 
                       model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       epoch: int,
                       metrics: Dict[str, float],
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                       is_best: bool = False,
                       filename: Optional[str] = None):
        """
        Save model checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            metrics: Dictionary of metrics
            scheduler: Learning rate scheduler (optional)
            is_best: Whether this is the best model so far
            filename: Custom filename (default: checkpoint_epoch_{epoch}.pth)
        """
        if filename is None:
            filename = f"checkpoint_epoch_{epoch}.pth"
        
        checkpoint_path = self.checkpoint_dir / filename
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model separately
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")
        
        # Manage checkpoint history
        self.checkpoint_history.append(checkpoint_path)
        if len(self.checkpoint_history) > self.max_checkpoints:
            old_checkpoint = self.checkpoint_history.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()
        
        return checkpoint_path
    
    def load_checkpoint(self, 
                       checkpoint_path: str,
                       model: torch.nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                       device: torch.device = torch.device('cpu')) -> Dict[str, Any]:
        """
        Load checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            device: Device to map checkpoint to
        
        Returns:
            Dictionary containing epoch and metrics
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        
        return {
            'epoch': checkpoint['epoch'],
            'metrics': checkpoint.get('metrics', {})
        }
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to the latest checkpoint."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pth"))
        if not checkpoints:
            return None
        return max(checkpoints, key=lambda p: p.stat().st_mtime)
    
    def get_best_checkpoint(self) -> Optional[Path]:
        """Get path to the best checkpoint."""
        best_path = self.checkpoint_dir / "best_model.pth"
        return best_path if best_path.exists() else None
```

### 1.5 Logging System

**CLAUDE CODE TASK**: Implement experiment logging and tracking.

```python
# File: src/experiments/logger.py
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from torch.utils.tensorboard import SummaryWriter

class ExperimentLogger:
    """Logger for experiment tracking and results."""
    
    def __init__(self, 
                 experiment_name: str,
                 log_dir: str = './results/logs',
                 use_tensorboard: bool = True,
                 tensorboard_dir: str = './results/tensorboard'):
        """
        Initialize experiment logger.
        
        Args:
            experiment_name: Name of the experiment
            log_dir: Directory for log files
            use_tensorboard: Whether to use TensorBoard
            tensorboard_dir: Directory for TensorBoard logs
        """
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_name = f"{experiment_name}_{self.timestamp}"
        
        # Create directories
        self.log_dir = Path(log_dir) / self.run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize log file
        self.log_file = self.log_dir / 'experiment.log'
        
        # Initialize TensorBoard
        self.use_tensorboard = use_tensorboard
        if use_tensorboard:
            tb_dir = Path(tensorboard_dir) / self.run_name
            self.writer = SummaryWriter(log_dir=str(tb_dir))
        else:
            self.writer = None
        
        # Initialize metrics history
        self.metrics_history = {
            'train': [],
            'val': [],
            'test': []
        }
        
        self.log_message(f"Experiment initialized: {self.run_name}")
    
    def log_message(self, message: str, level: str = 'INFO'):
        """Log a text message."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] [{level}] {message}"
        
        # Print to console
        print(log_entry)
        
        # Write to file
        with open(self.log_file, 'a') as f:
            f.write(log_entry + '\n')
    
    def log_metrics(self, 
                   metrics: Dict[str, float],
                   step: int,
                   phase: str = 'train'):
        """
        Log metrics for current step.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Current step/epoch number
            phase: 'train', 'val', or 'test'
        """
        # Add to history
        metrics_with_step = {'step': step, **metrics}
        self.metrics_history[phase].append(metrics_with_step)
        
        # Log to TensorBoard
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(f"{phase}/{key}", value, step)
        
        # Log message
        metrics_str = ', '.join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.log_message(f"[{phase.upper()}] Step {step}: {metrics_str}")
    
    def log_hyperparameters(self, config: Dict[str, Any]):
        """Log hyperparameters."""
        config_file = self.log_dir / 'config.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        if self.writer:
            # Log hyperparameters to TensorBoard
            self.writer.add_text('hyperparameters', json.dumps(config, indent=2), 0)
    
    def save_metrics_history(self):
        """Save complete metrics history to JSON."""
        metrics_file = self.log_dir / 'metrics_history.json'
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def close(self):
        """Close logger and save final results."""
        self.save_metrics_history()
        if self.writer:
            self.writer.close()
        self.log_message("Experiment logging complete")
```

### Phase 1 Validation Checkpoint

**CLAUDE CODE TASK**: Create and run infrastructure tests.

```python
# File: tests/test_utils/test_config.py
import pytest
from src.utils.config import Config

def test_config_default():
    """Test default configuration loading."""
    config = Config()
    assert config.get('model.encoder_arch') == 'resnet18'
    assert config.get('training.batch_size') == 128
    assert config.get('data.dataset') == 'cifar10'

def test_config_get_nested():
    """Test nested key access."""
    config = Config()
    assert config.get('model.embedding_dim') == 128
    assert config.get('nonexistent.key', 'default') == 'default'

def test_config_update():
    """Test configuration updates."""
    config = Config()
    config.update({'model': {'embedding_dim': 256}})
    assert config.get('model.embedding_dim') == 256
```

```python
# File: tests/test_utils/test_device_manager.py
import pytest
import torch
from src.utils.device_manager import DeviceManager

def test_device_manager_cpu():
    """Test CPU device manager."""
    dm = DeviceManager(device='cpu')
    assert dm.device.type == 'cpu'

def test_device_to_device():
    """Test moving tensors to device."""
    dm = DeviceManager(device='cpu')
    tensor = torch.randn(10, 10)
    tensor_device = dm.to_device(tensor)
    assert tensor_device.device.type == 'cpu'
```

**Run tests:**
```bash
pytest tests/test_utils/ -v
```

---

## Phase 2: Data Pipeline (3-4 hours)

### 2.1 Data Transformations

**CLAUDE CODE TASK**: Implement data augmentation and preprocessing.

```python
# File: src/data/transforms.py
import torch
import torchvision.transforms as transforms
from typing import List, Tuple, Optional

def get_train_transforms(config: dict) -> transforms.Compose:
    """
    Get training data transformations.
    
    Args:
        config: Augmentation configuration dictionary
    
    Returns:
        Composed transforms for training data
    """
    transform_list = []
    
    # Random crop with padding
    if config.get('random_crop', False):
        padding = config.get('crop_padding', 4)
        transform_list.append(transforms.RandomCrop(32, padding=padding))
    
    # Random horizontal flip
    if config.get('random_horizontal_flip', False):
        flip_prob = config.get('flip_probability', 0.5)
        transform_list.append(transforms.RandomHorizontalFlip(p=flip_prob))
    
    # Color jitter
    if config.get('color_jitter', False):
        jitter_params = config.get('jitter_params', {})
        transform_list.append(transforms.ColorJitter(
            brightness=jitter_params.get('brightness', 0.2),
            contrast=jitter_params.get('contrast', 0.2),
            saturation=jitter_params.get('saturation', 0.2),
            hue=jitter_params.get('hue', 0.1)
        ))
    
    # Random rotation
    if config.get('random_rotation', False):
        rotation_degrees = config.get('rotation_degrees', 15)
        transform_list.append(transforms.RandomRotation(rotation_degrees))
    
    # Convert to tensor
    transform_list.append(transforms.ToTensor())
    
    # Normalization
    if config.get('normalize', False):
        mean = config.get('normalization_mean', [0.5, 0.5, 0.5])
        std = config.get('normalization_std', [0.5, 0.5, 0.5])
        transform_list.append(transforms.Normalize(mean=mean, std=std))
    
    return transforms.Compose(transform_list)


def get_test_transforms(config: dict) -> transforms.Compose:
    """
    Get test data transformations.
    
    Args:
        config: Augmentation configuration dictionary
    
    Returns:
        Composed transforms for test data
    """
    transform_list = [transforms.ToTensor()]
    
    # Normalization
    if config.get('normalize', False):
        mean = config.get('normalization_mean', [0.5, 0.5, 0.5])
        std = config.get('normalization_std', [0.5, 0.5, 0.5])
        transform_list.append(transforms.Normalize(mean=mean, std=std))
    
    return transforms.Compose(transform_list)


class StrongAugmentation:
    """Strong augmentation for background class generation."""
    
    def __init__(self, augmentation_strength: float = 0.8):
        """
        Args:
            augmentation_strength: Strength of augmentation [0, 1]
        """
        self.strength = augmentation_strength
        
        self.transform = transforms.Compose([
            transforms.RandomCrop(32, padding=8),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=30 * augmentation_strength),
            transforms.ColorJitter(
                brightness=0.4 * augmentation_strength,
                contrast=0.4 * augmentation_strength,
                saturation=0.4 * augmentation_strength,
                hue=0.2 * augmentation_strength
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def __call__(self, img):
        return self.transform(img)
```

### 2.2 Background Class Generation

**CLAUDE CODE TASK**: Implement background/outlier sample generation.

```python
# File: src/data/background_generator.py
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple
from PIL import Image

class BackgroundGenerator:
    """
    Generates background/outlier samples for training.
    
    Strategy: Use samples from unknown classes with strong augmentation
    to create OOD (out-of-distribution) examples.
    """
    
    def __init__(self, 
                 source_dataset: Dataset,
                 known_classes: List[int],
                 augmentation_strength: float = 0.8):
        """
        Args:
            source_dataset: Dataset to sample from
            known_classes: List of known class indices to exclude
            augmentation_strength: Strength of augmentation [0, 1]
        """
        self.source_dataset = source_dataset
        self.known_classes = set(known_classes)
        self.augmentation_strength = augmentation_strength
        
        # Create strong augmentation transform
        self.strong_augment = transforms.Compose([
            transforms.ToPILImage() if not isinstance(source_dataset[0][0], Image.Image) else transforms.Lambda(lambda x: x),
            transforms.RandomCrop(32, padding=8),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=int(30 * augmentation_strength)),
            transforms.ColorJitter(
                brightness=0.5 * augmentation_strength,
                contrast=0.5 * augmentation_strength,
                saturation=0.5 * augmentation_strength,
                hue=0.3 * augmentation_strength
            ),
            transforms.RandomGrayscale(p=0.3),
            transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.3),
            transforms.ToTensor(),
        ])
        
        # Build indices of unknown class samples
        self.unknown_indices = []
        for idx in range(len(source_dataset)):
            _, label = source_dataset[idx]
            if label not in self.known_classes:
                self.unknown_indices.append(idx)
        
        print(f"Background generator: {len(self.unknown_indices)} unknown samples available")
    
    def generate_background_batch(self, batch_size: int) -> torch.Tensor:
        """
        Generate batch of background samples.
        
        Args:
            batch_size: Number of samples to generate
        
        Returns:
            Batch of background images, shape (batch_size, C, H, W)
        """
        # Randomly sample from unknown classes
        sampled_indices = np.random.choice(
            self.unknown_indices, 
            size=batch_size, 
            replace=True
        )
        
        background_samples = []
        for idx in sampled_indices:
            img, _ = self.source_dataset[idx]
            augmented_img = self.strong_augment(img)
            background_samples.append(augmented_img)
        
        return torch.stack(background_samples)
    
    def get_background_label(self) -> int:
        """
        Get the label index for background class.
        
        Returns:
            Background class label (highest class index + 1)
        """
        max_class = max(self.known_classes)
        return max_class + 1


class BackgroundAugmentedDataset(Dataset):
    """Dataset that includes background class samples."""
    
    def __init__(self, 
                 base_dataset: Dataset,
                 background_generator: BackgroundGenerator,
                 background_ratio: float = 0.3):
        """
        Args:
            base_dataset: Original dataset with known classes
            background_generator: Background sample generator
            background_ratio: Ratio of background samples (0 to 1)
        """
        self.base_dataset = base_dataset
        self.background_generator = background_generator
        self.background_ratio = background_ratio
        
        # Calculate number of background samples
        self.num_base = len(base_dataset)
        self.num_background = int(self.num_base * background_ratio / (1 - background_ratio))
        self.total_length = self.num_base + self.num_background
        
        self.background_label = background_generator.get_background_label()
    
    def __len__(self) -> int:
        return self.total_length
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get item by index.
        
        Returns:
            Tuple of (image, label)
        """
        if idx < self.num_base:
            # Return from base dataset
            return self.base_dataset[idx]
        else:
            # Generate background sample
            bg_img = self.background_generator.generate_background_batch(1)[0]
            return bg_img, self.background_label
```

### 2.3 Open-Set Data Splitting

**CLAUDE CODE TASK**: Implement open-set train/test split logic.

```python
# File: src/data/openset_split.py
import numpy as np
from typing import List, Tuple, Dict
import json
from pathlib import Path

class OpenSetSplitter:
    """Creates open-set train/test splits from datasets."""
    
    def __init__(self, 
                 num_total_classes: int,
                 num_known_classes: int,
                 known_class_indices: List[int] = None,
                 seed: int = 42):
        """
        Args:
            num_total_classes: Total number of classes in dataset
            num_known_classes: Number of known classes for training
            known_class_indices: Specific indices for known classes (None for random)
            seed: Random seed for reproducibility
        """
        self.num_total_classes = num_total_classes
        self.num_known_classes = num_known_classes
        self.seed = seed
        
        np.random.seed(seed)
        
        # Determine known vs unknown classes
        if known_class_indices is not None:
            self.known_classes = known_class_indices
        else:
            all_classes = list(range(num_total_classes))
            self.known_classes = sorted(np.random.choice(
                all_classes, 
                size=num_known_classes, 
                replace=False
            ).tolist())
        
        self.unknown_classes = [c for c in range(num_total_classes) 
                               if c not in self.known_classes]
        
        print(f"Known classes: {self.known_classes}")
        print(f"Unknown classes: {self.unknown_classes}")
    
    def split_dataset_indices(self, 
                              dataset_labels: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Split dataset indices into known/unknown.
        
        Args:
            dataset_labels: Array of labels for entire dataset
        
        Returns:
            Dictionary with 'train_known', 'test_known', 'test_unknown' indices
        """
        known_mask = np.isin(dataset_labels, self.known_classes)
        unknown_mask = ~known_mask
        
        known_indices = np.where(known_mask)[0]
        unknown_indices = np.where(unknown_mask)[0]
        
        # Split known indices into train/test (80/20)
        np.random.shuffle(known_indices)
        split_point = int(0.8 * len(known_indices))
        train_known_indices = known_indices[:split_point]
        test_known_indices = known_indices[split_point:]
        
        return {
            'train_known': train_known_indices,
            'test_known': test_known_indices,
            'test_unknown': unknown_indices
        }
    
    def save_split(self, split_dict: Dict[str, np.ndarray], save_path: str):
        """Save split indices to file."""
        save_dict = {k: v.tolist() for k, v in split_dict.items()}
        save_dict['known_classes'] = self.known_classes
        save_dict['unknown_classes'] = self.unknown_classes
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(save_dict, f, indent=2)
    
    @staticmethod
    def load_split(load_path: str) -> Dict[str, np.ndarray]:
        """Load split indices from file."""
        with open(load_path, 'r') as f:
            split_dict = json.load(f)
        
        return {
            'train_known': np.array(split_dict['train_known']),
            'test_known': np.array(split_dict['test_known']),
            'test_unknown': np.array(split_dict['test_unknown']),
            'known_classes': split_dict['known_classes'],
            'unknown_classes': split_dict['unknown_classes']
        }
```

### 2.4 Dataset Loaders

**CLAUDE CODE TASK**: Implement dataset loading for all supported datasets.

```python
# File: src/data/dataset_loader.py
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.datasets as datasets
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from .transforms import get_train_transforms, get_test_transforms
from .openset_split import OpenSetSplitter
from .background_generator import BackgroundGenerator, BackgroundAugmentedDataset


class OpenSetDataset(Dataset):
    """Wrapper dataset for open-set recognition."""
    
    def __init__(self, 
                 base_dataset: Dataset,
                 indices: np.ndarray,
                 known_classes: List[int]):
        """
        Args:
            base_dataset: Original PyTorch dataset
            indices: Indices to include in this split
            known_classes: List of known class indices
        """
        self.base_dataset = base_dataset
        self.indices = indices
        self.known_classes = set(known_classes)
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, bool]:
        """
        Returns:
            Tuple of (image, label, is_known)
        """
        real_idx = self.indices[idx]
        img, label = self.base_dataset[real_idx]
        is_known = label in self.known_classes
        return img, label, is_known


def load_base_dataset(dataset_name: str, 
                      split: str, 
                      transform,
                      data_root: str = './data') -> Dataset:
    """
    Load base dataset.
    
    Args:
        dataset_name: 'mnist', 'svhn', 'cifar10', 'cifar+10', 'cifar+50', 'tiny_imagenet'
        split: 'train' or 'test'
        transform: Data transformation
        data_root: Root directory for data
    
    Returns:
        PyTorch Dataset
    """
    is_train = (split == 'train')
    data_path = Path(data_root) / 'raw'
    data_path.mkdir(parents=True, exist_ok=True)
    
    if dataset_name == 'mnist':
        dataset = datasets.MNIST(
            root=str(data_path),
            train=is_train,
            transform=transform,
            download=True
        )
    
    elif dataset_name == 'svhn':
        split_name = 'train' if is_train else 'test'
        dataset = datasets.SVHN(
            root=str(data_path),
            split=split_name,
            transform=transform,
            download=True
        )
    
    elif dataset_name == 'cifar10':
        dataset = datasets.CIFAR10(
            root=str(data_path),
            train=is_train,
            transform=transform,
            download=True
        )
    
    elif dataset_name == 'cifar100':
        dataset = datasets.CIFAR100(
            root=str(data_path),
            train=is_train,
            transform=transform,
            download=True
        )
    
    elif dataset_name in ['cifar+10', 'cifar+50']:
        # CIFAR+10: CIFAR-10 as known, 10 CIFAR-100 classes as unknown
        # CIFAR+50: CIFAR-10 as known, 50 CIFAR-100 classes as unknown
        # We load CIFAR-10 and CIFAR-100, then combine them appropriately
        cifar10 = datasets.CIFAR10(
            root=str(data_path),
            train=is_train,
            transform=transform,
            download=True
        )
        cifar100 = datasets.CIFAR100(
            root=str(data_path),
            train=is_train,
            transform=transform,
            download=True
        )
        # Combine datasets (implementation simplified)
        dataset = cifar10  # Primary dataset for known classes
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset


def get_dataset_info(dataset_name: str) -> Dict:
    """
    Get dataset metadata.
    
    Returns:
        Dictionary with image_size, channels, num_classes
    """
    info = {
        'mnist': {'image_size': 28, 'channels': 1, 'num_classes': 10},
        'svhn': {'image_size': 32, 'channels': 3, 'num_classes': 10},
        'cifar10': {'image_size': 32, 'channels': 3, 'num_classes': 10},
        'cifar100': {'image_size': 32, 'channels': 3, 'num_classes': 100},
        'cifar+10': {'image_size': 32, 'channels': 3, 'num_classes': 20},
        'cifar+50': {'image_size': 32, 'channels': 3, 'num_classes': 60},
        'tiny_imagenet': {'image_size': 64, 'channels': 3, 'num_classes': 200},
    }
    return info.get(dataset_name, {'image_size': 32, 'channels': 3, 'num_classes': 10})


def get_openset_loaders(dataset_name: str,
                       known_classes: List[int],
                       batch_size: int,
                       num_workers: int = 4,
                       data_root: str = './data',
                       config: dict = None,
                       use_background: bool = True,
                       background_ratio: float = 0.3) -> Dict[str, DataLoader]:
    """
    Create data loaders for open-set experiments.
    
    Args:
        dataset_name: Name of dataset
        known_classes: List of known class indices
        batch_size: Batch size
        num_workers: Number of data loading workers
        data_root: Root directory for data
        config: Configuration dictionary
        use_background: Whether to include background class
        background_ratio: Ratio of background samples
    
    Returns:
        Dictionary with 'train', 'test_known', 'test_unknown' DataLoaders
    """
    if config is None:
        config = {}
    
    # Get transforms
    train_transform = get_train_transforms(config.get('train', {}))
    test_transform = get_test_transforms(config.get('test', {}))
    
    # Load base datasets
    train_dataset = load_base_dataset(dataset_name, 'train', train_transform, data_root)
    test_dataset = load_base_dataset(dataset_name, 'test', test_transform, data_root)
    
    # Get dataset info
    dataset_info = get_dataset_info(dataset_name)
    num_total_classes = dataset_info['num_classes']
    
    # Create open-set split
    splitter = OpenSetSplitter(
        num_total_classes=num_total_classes,
        num_known_classes=len(known_classes),
        known_class_indices=known_classes
    )
    
    # Get labels for splitting
    if hasattr(train_dataset, 'targets'):
        train_labels = np.array(train_dataset.targets)
    elif hasattr(train_dataset, 'labels'):
        train_labels = np.array(train_dataset.labels)
    else:
        train_labels = np.array([train_dataset[i][1] for i in range(len(train_dataset))])
    
    # Split train dataset
    train_split = splitter.split_dataset_indices(train_labels)
    
    # Create train dataset with known classes
    train_known_dataset = OpenSetDataset(
        base_dataset=train_dataset,
        indices=train_split['train_known'],
        known_classes=known_classes
    )
    
    # Add background class if requested
    if use_background:
        # Create background generator
        bg_generator = BackgroundGenerator(
            source_dataset=train_dataset,
            known_classes=known_classes,
            augmentation_strength=0.8
        )
        
        # Wrap with background augmentation
        train_known_dataset = BackgroundAugmentedDataset(
            base_dataset=train_known_dataset,
            background_generator=bg_generator,
            background_ratio=background_ratio
        )
    
    # Create test datasets
    if hasattr(test_dataset, 'targets'):
        test_labels = np.array(test_dataset.targets)
    elif hasattr(test_dataset, 'labels'):
        test_labels = np.array(test_dataset.labels)
    else:
        test_labels = np.array([test_dataset[i][1] for i in range(len(test_dataset))])
    
    test_split = splitter.split_dataset_indices(test_labels)
    
    test_known_dataset = OpenSetDataset(
        base_dataset=test_dataset,
        indices=test_split['test_known'],
        known_classes=known_classes
    )
    
    test_unknown_dataset = OpenSetDataset(
        base_dataset=test_dataset,
        indices=test_split['test_unknown'],
        known_classes=known_classes
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_known_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    test_known_loader = DataLoader(
        test_known_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_unknown_loader = DataLoader(
        test_unknown_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Data loaders created:")
    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Test Known: {len(test_known_loader.dataset)} samples")
    print(f"  Test Unknown: {len(test_unknown_loader.dataset)} samples")
    
    return {
        'train': train_loader,
        'test_known': test_known_loader,
        'test_unknown': test_unknown_loader
    }
```

### Phase 2 Validation Checkpoint

**CLAUDE CODE TASK**: Test data pipeline.

```python
# File: tests/test_data/test_dataset_loader.py
import pytest
import torch
from src.data.dataset_loader import get_openset_loaders, get_dataset_info

def test_get_dataset_info():
    """Test dataset info retrieval."""
    info = get_dataset_info('cifar10')
    assert info['image_size'] == 32
    assert info['channels'] == 3
    assert info['num_classes'] == 10

def test_openset_loaders_creation():
    """Test creation of open-set data loaders."""
    loaders = get_openset_loaders(
        dataset_name='cifar10',
        known_classes=[0, 1, 2, 3, 4, 5],
        batch_size=32,
        num_workers=0,
        use_background=False
    )
    
    assert 'train' in loaders
    assert 'test_known' in loaders
    assert 'test_unknown' in loaders
    
    # Test batch shape
    batch = next(iter(loaders['train']))
    assert len(batch) == 3  # (images, labels, is_known)
    images, labels, is_known = batch
    assert images.shape[0] == 32  # batch size
    assert images.shape[1] == 3   # channels

def test_background_augmentation():
    """Test background class generation."""
    loaders = get_openset_loaders(
        dataset_name='cifar10',
        known_classes=[0, 1, 2],
        batch_size=32,
        num_workers=0,
        use_background=True,
        background_ratio=0.3
    )
    
    # Check that background class label exists
    batch = next(iter(loaders['train']))
    images, labels, is_known = batch
    # Background label should be max(known_classes) + 1 = 3
    assert 3 in labels  # background class
```

**Run tests:**
```bash
pytest tests/test_data/ -v
```

---

## Phase 3: Model Implementation (4-5 hours)

### 3.1 FiLM Layers

**CLAUDE CODE TASK**: Implement Feature-wise Linear Modulation.

```python
# File: src/models/film_layers.py
import torch
import torch.nn as nn

class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation layer.
    
    Applies affine transformation: FiLM(F) = gamma * F + beta
    where gamma and beta are learned from class embeddings.
    """
    
    def __init__(self, num_features: int, embedding_dim: int):
        """
        Args:
            num_features: Number of input feature channels
            embedding_dim: Dimension of class embedding
        """
        super(FiLM, self).__init__()
        
        self.num_features = num_features
        self.embedding_dim = embedding_dim
        
        # Generate gamma (scale) and beta (shift) from embedding
        self.film_generator = FiLMGenerator(embedding_dim, num_features)
    
    def forward(self, features: torch.Tensor, class_embedding: torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM transformation.
        
        Args:
            features: Feature maps, shape (B, C, H, W)
            class_embedding: Class embeddings, shape (B, embedding_dim)
        
        Returns:
            Modulated features, shape (B, C, H, W)
        """
        gamma, beta = self.film_generator(class_embedding)
        
        # Apply affine transformation
        # gamma and beta have shape (B, C, 1, 1) for broadcasting
        return gamma * features + beta


class FiLMGenerator(nn.Module):
    """Generates FiLM parameters (gamma, beta) from class embeddings."""
    
    def __init__(self, embedding_dim: int, num_features: int, hidden_dim: int = 256):
        """
        Args:
            embedding_dim: Dimension of class embedding
            num_features: Number of feature channels to modulate
            hidden_dim: Hidden layer dimension
        """
        super(FiLMGenerator, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_features = num_features
        
        # Network to generate gamma (scale parameter)
        self.gamma_network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_features)
        )
        
        # Network to generate beta (shift parameter)
        self.beta_network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_features)
        )
        
        # Initialize gamma to 1 and beta to 0 for identity initialization
        nn.init.constant_(self.gamma_network[-1].weight, 0)
        nn.init.constant_(self.gamma_network[-1].bias, 1)
        nn.init.constant_(self.beta_network[-1].weight, 0)
        nn.init.constant_(self.beta_network[-1].bias, 0)
    
    def forward(self, class_embedding: torch.Tensor) -> tuple:
        """
        Generate FiLM parameters.
        
        Args:
            class_embedding: shape (B, embedding_dim)
        
        Returns:
            Tuple of (gamma, beta):
                - gamma: shape (B, num_features, 1, 1)
                - beta: shape (B, num_features, 1, 1)
        """
        gamma = self.gamma_network(class_embedding)
        beta = self.beta_network(class_embedding)
        
        # Reshape for broadcasting with feature maps (B, C, H, W)
        gamma = gamma.view(-1, self.num_features, 1, 1)
        beta = beta.view(-1, self.num_features, 1, 1)
        
        return gamma, beta
```

### 3.2 Class Embedding Layer

**CLAUDE CODE TASK**: Implement class embedding layer.

```python
# File: src/models/class_embedding.py
import torch
import torch.nn as nn

class ClassEmbedding(nn.Module):
    """Learnable class embedding layer."""
    
    def __init__(self, num_classes: int, embedding_dim: int):
        """
        Args:
            num_classes: Number of classes (including background)
            embedding_dim: Dimension of embedding vectors
        """
        super(ClassEmbedding, self).__init__()
        
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        
        # Learnable embedding matrix
        self.embedding = nn.Embedding(num_classes, embedding_dim)
        
        # Initialize with normal distribution
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)
    
    def forward(self, class_labels: torch.Tensor) -> torch.Tensor:
        """
        Get embeddings for class labels.
        
        Args:
            class_labels: Class indices, shape (B,)
        
        Returns:
            Class embeddings, shape (B, embedding_dim)
        """
        return self.embedding(class_labels)
    
    def get_all_embeddings(self) -> torch.Tensor:
        """
        Get all class embeddings.
        
        Returns:
            All embeddings, shape (num_classes, embedding_dim)
        """
        indices = torch.arange(self.num_classes, device=self.embedding.weight.device)
        return self.embedding(indices)
```

### 3.3 Encoder Implementation

**CLAUDE CODE TASK**: Implement encoder network.

```python
# File: src/models/encoder.py
import torch
import torch.nn as nn
import torchvision.models as models

class Encoder(nn.Module):
    """Encoder network for feature extraction."""
    
    def __init__(self, 
                 image_channels: int,
                 image_size: int,
                 latent_dim: int,
                 architecture: str = 'resnet18'):
        """
        Args:
            image_channels: Number of input channels (1 for grayscale, 3 for RGB)
            image_size: Input image size (assumes square images)
            latent_dim: Output latent dimension
            architecture: 'resnet18', 'resnet34', or 'custom'
        """
        super(Encoder, self).__init__()
        
        self.image_channels = image_channels
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.architecture = architecture
        
        if architecture == 'resnet18':
            self.encoder = self._build_resnet18_encoder()
        elif architecture == 'resnet34':
            self.encoder = self._build_resnet34_encoder()
        elif architecture == 'custom':
            self.encoder = self._build_custom_encoder()
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
    
    def _build_resnet18_encoder(self) -> nn.Module:
        """Build ResNet18-based encoder."""
        # Load pretrained ResNet18
        resnet = models.resnet18(pretrained=False)
        
        # Modify first conv layer if needed for different input channels
        if self.image_channels != 3:
            resnet.conv1 = nn.Conv2d(
                self.image_channels, 64, 
                kernel_size=7, stride=2, padding=3, bias=False
            )
        
        # Remove final FC layer
        modules = list(resnet.children())[:-1]  # Remove avgpool and fc
        encoder = nn.Sequential(*modules)
        
        # Add adaptive pooling and projection to latent_dim
        encoder.add_module('adaptive_pool', nn.AdaptiveAvgPool2d((1, 1)))
        encoder.add_module('flatten', nn.Flatten())
        encoder.add_module('projection', nn.Linear(512, self.latent_dim))
        
        return encoder
    
    def _build_resnet34_encoder(self) -> nn.Module:
        """Build ResNet34-based encoder."""
        resnet = models.resnet34(pretrained=False)
        
        if self.image_channels != 3:
            resnet.conv1 = nn.Conv2d(
                self.image_channels, 64,
                kernel_size=7, stride=2, padding=3, bias=False
            )
        
        modules = list(resnet.children())[:-1]
        encoder = nn.Sequential(*modules)
        encoder.add_module('adaptive_pool', nn.AdaptiveAvgPool2d((1, 1)))
        encoder.add_module('flatten', nn.Flatten())
        encoder.add_module('projection', nn.Linear(512, self.latent_dim))
        
        return encoder
    
    def _build_custom_encoder(self) -> nn.Module:
        """Build custom CNN encoder for smaller images (e.g., MNIST, CIFAR)."""
        layers = []
        
        # Conv block 1: channels -> 64
        layers.extend([
            nn.Conv2d(self.image_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # H/2, W/2
        ])
        
        # Conv block 2: 64 -> 128
        layers.extend([
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # H/4, W/4
        ])
        
        # Conv block 3: 128 -> 256
        layers.extend([
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # H/8, W/8
        ])
        
        # Conv block 4: 256 -> 512
        layers.extend([
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        ])
        
        # Flatten and project to latent dimension
        layers.extend([
            nn.Flatten(),
            nn.Linear(512, self.latent_dim),
            nn.ReLU(inplace=True)
        ])
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images, shape (B, C, H, W)
        
        Returns:
            Latent representation, shape (B, latent_dim)
        """
        return self.encoder(x)
```

### 3.4 Decoder Implementation

**CLAUDE CODE TASK**: Implement class-conditioned decoder with FiLM layers.

```python
# File: src/models/decoder.py
import torch
import torch.nn as nn
from .film_layers import FiLM

class ConditionalDecoder(nn.Module):
    """Decoder with FiLM conditioning at multiple layers."""
    
    def __init__(self,
                 latent_dim: int,
                 embedding_dim: int,
                 image_channels: int,
                 image_size: int,
                 film_positions: list = [0, 1, 2, 3]):
        """
        Args:
            latent_dim: Dimension of latent input
            embedding_dim: Dimension of class embedding
            image_channels: Number of output image channels
            image_size: Output image size
            film_positions: Layer indices where FiLM is applied
        """
        super(ConditionalDecoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.image_channels = image_channels
        self.image_size = image_size
        self.film_positions = film_positions
        
        # Calculate initial spatial size after projection
        self.initial_size = image_size // 8  # After 3 upsampling layers
        
        # Project latent to spatial features
        self.fc = nn.Linear(latent_dim, 512 * self.initial_size * self.initial_size)
        
        # Decoder blocks with FiLM conditioning
        self.decoder_blocks = nn.ModuleList()
        self.film_layers = nn.ModuleList()
        
        # Block 0: 512 -> 256
        self.decoder_blocks.append(self._make_decoder_block(512, 256))
        if 0 in film_positions:
            self.film_layers.append(FiLM(256, embedding_dim))
        else:
            self.film_layers.append(None)
        
        # Block 1: 256 -> 128
        self.decoder_blocks.append(self._make_decoder_block(256, 128))
        if 1 in film_positions:
            self.film_layers.append(FiLM(128, embedding_dim))
        else:
            self.film_layers.append(None)
        
        # Block 2: 128 -> 64
        self.decoder_blocks.append(self._make_decoder_block(128, 64))
        if 2 in film_positions:
            self.film_layers.append(FiLM(64, embedding_dim))
        else:
            self.film_layers.append(None)
        
        # Block 3: 64 -> 32
        self.decoder_blocks.append(self._make_decoder_block(64, 32))
        if 3 in film_positions:
            self.film_layers.append(FiLM(32, embedding_dim))
        else:
            self.film_layers.append(None)
        
        # Final convolution to output channels
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, image_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Output in range [-1, 1]
        )
    
    def _make_decoder_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Create a decoder block with upsampling."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 
                             kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, latent: torch.Tensor, class_embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with class conditioning.
        
        Args:
            latent: Latent representation, shape (B, latent_dim)
            class_embedding: Class embeddings, shape (B, embedding_dim)
        
        Returns:
            Reconstructed images, shape (B, image_channels, image_size, image_size)
        """
        # Project latent to spatial features
        x = self.fc(latent)
        x = x.view(-1, 512, self.initial_size, self.initial_size)
        
        # Apply decoder blocks with FiLM conditioning
        for i, (decoder_block, film_layer) in enumerate(zip(self.decoder_blocks, self.film_layers)):
            x = decoder_block(x)
            
            # Apply FiLM if present at this position
            if film_layer is not None:
                x = film_layer(x, class_embedding)
        
        # Final convolution
        x = self.final_conv(x)
        
        # Resize to exact output size if needed
        if x.shape[-1] != self.image_size:
            x = nn.functional.interpolate(x, size=(self.image_size, self.image_size), 
                                         mode='bilinear', align_corners=False)
        
        return x
```

### 3.5 Complete C2AE Model

**CLAUDE CODE TASK**: Implement the complete C2AE model integrating all components.

```python
# File: src/models/c2ae.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from .encoder import Encoder
from .decoder import ConditionalDecoder
from .class_embedding import ClassEmbedding

class C2AE(nn.Module):
    """Class Conditioned Auto-Encoder for Open-Set Recognition."""
    
    def __init__(self,
                 num_classes: int,
                 embedding_dim: int = 128,
                 encoder_arch: str = 'resnet18',
                 latent_dim: int = 512,
                 image_channels: int = 3,
                 image_size: int = 32,
                 film_positions: List[int] = [0, 1, 2, 3]):
        """
        Args:
            num_classes: Number of known classes (+1 for background)
            embedding_dim: Dimension of class embedding
            encoder_arch: Encoder architecture
            latent_dim: Latent space dimension
            image_channels: Number of input channels
            image_size: Input image size
            film_positions: Decoder layers where FiLM is applied
        """
        super(C2AE, self).__init__()
        
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.image_channels = image_channels
        self.image_size = image_size
        
        # Class embedding layer
        self.class_embedding = ClassEmbedding(num_classes, embedding_dim)
        
        # Encoder
        self.encoder = Encoder(
            image_channels=image_channels,
            image_size=image_size,
            latent_dim=latent_dim,
            architecture=encoder_arch
        )
        
        # Decoder with class conditioning
        self.decoder = ConditionalDecoder(
            latent_dim=latent_dim,
            embedding_dim=embedding_dim,
            image_channels=image_channels,
            image_size=image_size,
            film_positions=film_positions
        )
    
    def forward(self, x: torch.Tensor, class_labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through C2AE.
        
        Args:
            x: Input images, shape (B, C, H, W)
            class_labels: Class labels for conditioning, shape (B,)
        
        Returns:
            Dictionary containing:
                - 'reconstructed': Reconstructed images
                - 'latent': Latent representations
                - 'class_embedding': Class embeddings used
        """
        # Encode input to latent space
        latent = self.encoder(x)
        
        # Get class embeddings
        class_emb = self.class_embedding(class_labels)
        
        # Decode with class conditioning
        reconstructed = self.decoder(latent, class_emb)
        
        return {
            'reconstructed': reconstructed,
            'latent': latent,
            'class_embedding': class_emb
        }
    
    def compute_reconstruction_error(self, 
                                    x: torch.Tensor, 
                                    reconstructed: torch.Tensor,
                                    reduction: str = 'none') -> torch.Tensor:
        """
        Compute pixel-wise reconstruction error.
        
        Args:
            x: Original images, shape (B, C, H, W)
            reconstructed: Reconstructed images, shape (B, C, H, W)
            reduction: 'none', 'mean', or 'sum'
        
        Returns:
            Reconstruction errors:
                - If reduction='none': shape (B, C, H, W)
                - If reduction='mean' or 'sum': shape (B,)
        """
        # Compute MSE
        error = (x - reconstructed) ** 2
        
        if reduction == 'none':
            return error
        elif reduction == 'mean':
            # Average over spatial and channel dimensions
            return error.view(error.size(0), -1).mean(dim=1)
        elif reduction == 'sum':
            return error.view(error.size(0), -1).sum(dim=1)
        else:
            raise ValueError(f"Unknown reduction: {reduction}")
    
    def predict_openset(self,
                       x: torch.Tensor,
                       known_classes: List[int],
                       threshold: Optional[float] = None,
                       return_scores: bool = True) -> Dict[str, torch.Tensor]:
        """
        Predict whether samples are known or unknown classes.
        
        Strategy: Try to reconstruct with each known class embedding.
        The class with minimum reconstruction error is the prediction.
        If all errors exceed threshold, classify as unknown.
        
        Args:
            x: Input images, shape (B, C, H, W)
            known_classes: List of known class indices
            threshold: Decision threshold (None for auto)
            return_scores: Whether to return reconstruction scores
        
        Returns:
            Dictionary containing:
                - 'predictions': Predicted class indices (-1 for unknown)
                - 'scores': Reconstruction errors for each class
                - 'is_known': Boolean mask for known predictions
        """
        batch_size = x.size(0)
        device = x.device
        
        # Encode once
        latent = self.encoder(x)
        
        # Try reconstructing with each known class
        all_errors = []
        
        for class_idx in known_classes:
            # Create class labels
            class_labels = torch.full((batch_size,), class_idx, 
                                     dtype=torch.long, device=device)
            
            # Get class embeddings
            class_emb = self.class_embedding(class_labels)
            
            # Reconstruct
            reconstructed = self.decoder(latent, class_emb)
            
            # Compute reconstruction error
            errors = self.compute_reconstruction_error(x, reconstructed, reduction='mean')
            all_errors.append(errors)
        
        # Stack errors: shape (B, num_known_classes)
        all_errors = torch.stack(all_errors, dim=1)
        
        # Find minimum error and corresponding class
        min_errors, min_indices = torch.min(all_errors, dim=1)
        
        # Map indices to actual class labels
        known_classes_tensor = torch.tensor(known_classes, device=device)
        predictions = known_classes_tensor[min_indices]
        
        # Determine if known or unknown
        if threshold is not None:
            is_known = min_errors < threshold
            predictions = torch.where(is_known, predictions, 
                                    torch.full_like(predictions, -1))
        else:
            is_known = torch.ones(batch_size, dtype=torch.bool, device=device)
        
        result = {
            'predictions': predictions,
            'is_known': is_known
        }
        
        if return_scores:
            result['scores'] = all_errors
            result['min_scores'] = min_errors
        
        return result
```

### Phase 3 Validation Checkpoint

**CLAUDE CODE TASK**: Test model components.

```python
# File: tests/test_models/test_c2ae.py
import pytest
import torch
from src.models.c2ae import C2AE

def test_c2ae_forward():
    """Test C2AE forward pass."""
    model = C2AE(
        num_classes=7,  # 6 known + 1 background
        embedding_dim=128,
        encoder_arch='custom',
        latent_dim=512,
        image_channels=3,
        image_size=32
    )
    
    batch_size = 8
    x = torch.randn(batch_size, 3, 32, 32)
    class_labels = torch.randint(0, 7, (batch_size,))
    
    output = model(x, class_labels)
    
    assert 'reconstructed' in output
    assert 'latent' in output
    assert 'class_embedding' in output
    
    assert output['reconstructed'].shape == (batch_size, 3, 32, 32)
    assert output['latent'].shape == (batch_size, 512)
    assert output['class_embedding'].shape == (batch_size, 128)

def test_reconstruction_error():
    """Test reconstruction error computation."""
    model = C2AE(num_classes=7, image_channels=3, image_size=32)
    
    x = torch.randn(4, 3, 32, 32)
    reconstructed = torch.randn(4, 3, 32, 32)
    
    error_none = model.compute_reconstruction_error(x, reconstructed, reduction='none')
    error_mean = model.compute_reconstruction_error(x, reconstructed, reduction='mean')
    
    assert error_none.shape == (4, 3, 32, 32)
    assert error_mean.shape == (4,)

def test_openset_prediction():
    """Test open-set prediction."""
    model = C2AE(num_classes=7, image_channels=3, image_size=32)
    model.eval()
    
    x = torch.randn(4, 3, 32, 32)
    known_classes = [0, 1, 2, 3, 4, 5]
    
    with torch.no_grad():
        result = model.predict_openset(x, known_classes, threshold=0.5)
    
    assert 'predictions' in result
    assert 'is_known' in result
    assert 'scores' in result
    assert result['predictions'].shape == (4,)
    assert result['is_known'].shape == (4,)
```

**Run tests:**
```bash
pytest tests/test_models/ -v
```

---

**(Continued in next message due to length...)**
