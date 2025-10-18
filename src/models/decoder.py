import torch
import torch.nn as nn
from .film_layers import FiLM


class ConditionalDecoder(nn.Module):
    """
    Decoder with FiLM conditioning at multiple layers.
    
    Args:
        latent_dim: Dimension of latent input
        embedding_dim: Dimension of class embedding
        image_channels: Number of output image channels
        image_size: Output image size
        film_positions: Layer indices where FiLM is applied
    """
    
    def __init__(self, 
                 latent_dim: int, 
                 embedding_dim: int, 
                 image_channels: int, 
                 image_size: int,
                 film_positions: list = [0, 1, 2, 3]):
        super(ConditionalDecoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.image_channels = image_channels
        self.image_size = image_size
        self.film_positions = film_positions
        
        # Calculate initial spatial size after projection
        # We'll upsample to image_size
        self.init_size = image_size // 8  # Will be upsampled 3 times (2x each)
        self.init_channels = 256
        
        # Project latent to initial feature map
        self.fc = nn.Linear(latent_dim, self.init_channels * self.init_size * self.init_size)
        
        # Decoder blocks with FiLM conditioning
        self.decoder_blocks = nn.ModuleList()
        self.film_layers = nn.ModuleDict()
        
        # Block 0: 256 channels
        self.decoder_blocks.append(nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        ))
        if 0 in film_positions:
            self.film_layers['0'] = FiLM(256, embedding_dim)
        
        # Block 1: 256 -> 128 channels, upsample
        self.decoder_blocks.append(nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        ))
        if 1 in film_positions:
            self.film_layers['1'] = FiLM(128, embedding_dim)
        
        # Block 2: 128 -> 64 channels, upsample
        self.decoder_blocks.append(nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        ))
        if 2 in film_positions:
            self.film_layers['2'] = FiLM(64, embedding_dim)
        
        # Block 3: 64 -> 32 channels, upsample
        self.decoder_blocks.append(nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        ))
        if 3 in film_positions:
            self.film_layers['3'] = FiLM(32, embedding_dim)
        
        # Final output layer
        self.output_layer = nn.Sequential(
            nn.Conv2d(32, image_channels, kernel_size=3, padding=1),
            nn.Tanh()  # Output in [-1, 1] range
        )
    
    def forward(self, latent: torch.Tensor, class_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: Latent representation, shape (batch_size, latent_dim)
            class_embedding: Class embeddings, shape (batch_size, embedding_dim)
        
        Returns:
            Reconstructed images, shape (batch_size, channels, H, W)
        """
        # Project and reshape to initial feature map
        x = self.fc(latent)
        x = x.view(-1, self.init_channels, self.init_size, self.init_size)
        
        # Pass through decoder blocks with FiLM conditioning
        for i, block in enumerate(self.decoder_blocks):
            x = block(x)
            
            # Apply FiLM if configured for this position
            if str(i) in self.film_layers:
                x = self.film_layers[str(i)](x, class_embedding)
        
        # Generate output image
        x = self.output_layer(x)
        
        # Ensure output size matches target
        if x.shape[-1] != self.image_size:
            x = torch.nn.functional.interpolate(x, size=(self.image_size, self.image_size), 
                                               mode='bilinear', align_corners=False)
        
        return x
