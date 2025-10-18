import torch
import torch.nn as nn
from typing import Dict, Optional, List
from .encoder import Encoder
from .decoder import ConditionalDecoder
from .class_embedding import ClassEmbedding


class C2AE(nn.Module):
    """
    Main Class Conditioned Auto-Encoder model.
    
    Args:
        num_classes: Number of known classes (+1 for background)
        embedding_dim: Dimension of class embedding
        encoder_arch: Encoder architecture ['resnet18', 'resnet34', 'custom']
        latent_dim: Dimension of latent representation
        image_channels: Number of input image channels
        image_size: Input image size (assumes square images)
        film_positions: Positions where FiLM layers are applied in decoder
    """
    
    def __init__(self, 
                 num_classes: int, 
                 embedding_dim: int = 128, 
                 encoder_arch: str = 'resnet18', 
                 latent_dim: int = 512, 
                 image_channels: int = 3, 
                 image_size: int = 32,
                 film_positions: List[int] = [0, 1, 2, 3]):
        super(C2AE, self).__init__()
        
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.encoder_arch = encoder_arch
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
        
        # Conditional decoder with FiLM layers
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
            x: Input images, shape (batch_size, channels, height, width)
            class_labels: Class labels for conditioning, shape (batch_size,)
        
        Returns:
            dict: {
                'reconstructed': torch.Tensor, shape (batch_size, channels, height, width)
                'latent': torch.Tensor, shape (batch_size, latent_dim)
                'class_embedding': torch.Tensor, shape (batch_size, embedding_dim)
            }
        """
        # Encode input to latent representation
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
            x: Original images
            reconstructed: Reconstructed images
            reduction: 'none', 'mean', 'sum'
        
        Returns:
            Reconstruction errors
        """
        # Compute MSE
        error = (x - reconstructed) ** 2
        
        if reduction == 'none':
            return error
        elif reduction == 'mean':
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
        
        Args:
            x: Input images
            known_classes: List of known class indices
            threshold: Decision threshold for unknown detection
            return_scores: Whether to return reconstruction scores
        
        Returns:
            dict: {
                'predictions': predicted class (-1 for unknown)
                'scores': reconstruction errors per class
                'is_known': boolean mask
            }
        """
        batch_size = x.size(0)
        device = x.device
        
        # Compute reconstruction error for each known class
        reconstruction_errors = torch.zeros(batch_size, len(known_classes), device=device)
        
        for i, class_idx in enumerate(known_classes):
            class_labels = torch.full((batch_size,), class_idx, dtype=torch.long, device=device)
            outputs = self.forward(x, class_labels)
            errors = self.compute_reconstruction_error(x, outputs['reconstructed'], reduction='mean')
            reconstruction_errors[:, i] = errors
        
        # Use minimum reconstruction error as score
        min_errors, best_classes = reconstruction_errors.min(dim=1)
        
        # Map back to actual class indices
        predictions = torch.tensor([known_classes[i] for i in best_classes], device=device)
        
        # Determine if known or unknown based on threshold
        if threshold is not None:
            is_known = min_errors < threshold
            predictions[~is_known] = -1  # Mark unknown as -1
        else:
            is_known = torch.ones(batch_size, dtype=torch.bool, device=device)
        
        result = {
            'predictions': predictions,
            'is_known': is_known,
            'min_reconstruction_error': min_errors
        }
        
        if return_scores:
            result['scores'] = reconstruction_errors
        
        return result
