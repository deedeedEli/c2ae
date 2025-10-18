import torch
import torch.nn as nn
from typing import Dict, Optional
from tqdm import tqdm
from .losses import C2AELoss
from ..utils.checkpoint import CheckpointManager
from ..experiments.logger import ExperimentLogger


class C2AETrainer:
    """
    Trainer for C2AE model.
    
    Args:
        model: C2AE model instance
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Training device
        config: Training configuration
        logger: Experiment logger
    """
    
    def __init__(self, 
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                 device: torch.device,
                 config: Dict,
                 logger: Optional[ExperimentLogger] = None):
        
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.logger = logger
        
        # Loss function
        self.criterion = C2AELoss(
            reconstruction_weight=config.get('training', {}).get('reconstruction_weight', 1.0)
        )
        
        # Checkpoint manager
        checkpoint_dir = config.get('experiment', {}).get('save_dir', './results') + '/checkpoints'
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
    def train_epoch(self, train_loader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        total_recon_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (images, labels, _) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images, labels)
            
            # Compute loss
            loss_dict = self.criterion(outputs, images)
            loss = loss_dict['total_loss']
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            total_recon_loss += loss_dict['reconstruction_loss'].item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'recon': loss_dict['reconstruction_loss'].item()
            })
        
        # Calculate averages
        avg_loss = total_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        
        metrics = {
            'loss': avg_loss,
            'reconstruction_loss': avg_recon_loss,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        return metrics
    
    def validate(self, val_loader) -> Dict[str, float]:
        """
        Validation step.
        
        Args:
            val_loader: Validation data loader
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_recon_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images, labels)
                
                # Compute loss
                loss_dict = self.criterion(outputs, images)
                
                # Track metrics
                total_loss += loss_dict['total_loss'].item()
                total_recon_loss += loss_dict['reconstruction_loss'].item()
                num_batches += 1
        
        # Calculate averages
        avg_loss = total_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        
        metrics = {
            'val_loss': avg_loss,
            'val_reconstruction_loss': avg_recon_loss
        }
        
        return metrics
    
    def fit(self, 
            train_loader, 
            val_loader, 
            num_epochs: int,
            checkpoint_dir: str = './results/checkpoints') -> Dict:
        """
        Full training loop.
        
        Args:
            train_loader: Training data
            val_loader: Validation data
            num_epochs: Number of training epochs
            checkpoint_dir: Directory to save checkpoints
        
        Returns:
            Training history
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        early_stopping_patience = self.config.get('training', {}).get('early_stopping_patience', 20)
        
        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch(train_loader, epoch)
            print(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_metrics['loss']:.4f}")
            
            # Validation
            val_metrics = self.validate(val_loader)
            print(f"Epoch {epoch}/{num_epochs} - Val Loss: {val_metrics['val_loss']:.4f}")
            
            # Update history
            history['train_loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['val_loss'])
            history['learning_rate'].append(train_metrics['learning_rate'])
            
            # Log metrics
            if self.logger is not None:
                self.logger.log_metrics(train_metrics, epoch, prefix='train')
                self.logger.log_metrics(val_metrics, epoch, prefix='val')
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Check if best model
            is_best = val_metrics['val_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['val_loss']
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            # Save checkpoint
            checkpoint_interval = self.config.get('experiment', {}).get('checkpoint_interval', 10)
            if epoch % checkpoint_interval == 0 or is_best:
                metrics = {**train_metrics, **val_metrics}
                self.checkpoint_manager.save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    metrics=metrics,
                    scheduler=self.scheduler,
                    is_best=is_best
                )
            
            # Early stopping
            if self.epochs_without_improvement >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break
        
        return history
