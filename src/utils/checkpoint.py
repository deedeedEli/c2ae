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
