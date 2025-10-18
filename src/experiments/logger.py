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
            tensorboard_dir: TensorBoard log directory
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create experiment-specific directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.log_dir / f"{experiment_name}_{timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize TensorBoard
        self.use_tensorboard = use_tensorboard
        if use_tensorboard:
            tb_dir = Path(tensorboard_dir) / f"{experiment_name}_{timestamp}"
            self.writer = SummaryWriter(log_dir=str(tb_dir))
        else:
            self.writer = None
        
        # Metrics storage
        self.metrics_history = []
    
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ''):
        """
        Log metrics.
        
        Args:
            metrics: Dictionary of metric name -> value
            step: Current step/epoch
            prefix: Prefix for metric names (e.g., 'train', 'val')
        """
        # Store in history
        log_entry = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }
        if prefix:
            log_entry['prefix'] = prefix
        self.metrics_history.append(log_entry)
        
        # Log to TensorBoard
        if self.writer is not None:
            for name, value in metrics.items():
                tag = f"{prefix}/{name}" if prefix else name
                self.writer.add_scalar(tag, value, step)
    
    def log_hyperparameters(self, hyperparams: Dict[str, Any]):
        """Log hyperparameters."""
        hp_path = self.experiment_dir / "hyperparameters.json"
        with open(hp_path, 'w') as f:
            json.dump(hyperparams, f, indent=2)
    
    def save_metrics(self):
        """Save all metrics to file."""
        metrics_path = self.experiment_dir / "metrics_history.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def close(self):
        """Close logger and save final results."""
        self.save_metrics()
        if self.writer is not None:
            self.writer.close()
