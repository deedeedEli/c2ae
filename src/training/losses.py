import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconstructionLoss(nn.Module):
    """
    Reconstruction loss for C2AE.
    
    Args:
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(self, reduction: str = 'mean'):
        super(ReconstructionLoss, self).__init__()
        self.reduction = reduction
        self.mse = nn.MSELoss(reduction=reduction)
    
    def forward(self, reconstructed: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction loss.
        
        Args:
            reconstructed: Reconstructed images
            target: Original images
        
        Returns:
            Loss value
        """
        return self.mse(reconstructed, target)


class C2AELoss(nn.Module):
    """
    Combined loss for C2AE training.
    
    Args:
        reconstruction_weight: Weight for reconstruction loss
    """
    
    def __init__(self, reconstruction_weight: float = 1.0):
        super(C2AELoss, self).__init__()
        self.reconstruction_weight = reconstruction_weight
        self.reconstruction_loss = ReconstructionLoss(reduction='mean')
    
    def forward(self, outputs: dict, targets: torch.Tensor) -> dict:
        """
        Compute combined loss.
        
        Args:
            outputs: Model outputs containing 'reconstructed', 'latent', 'class_embedding'
            targets: Target images
        
        Returns:
            Dictionary containing:
                - 'total_loss': Total loss value
                - 'reconstruction_loss': Reconstruction loss component
        """
        reconstructed = outputs['reconstructed']
        
        # Reconstruction loss
        recon_loss = self.reconstruction_loss(reconstructed, targets)
        
        # Total loss (can add regularization terms here if needed)
        total_loss = self.reconstruction_weight * recon_loss
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': recon_loss
        }
