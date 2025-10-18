import torch
import torch.nn as nn
import torchvision.models as models


class Encoder(nn.Module):
    """
    CNN-based encoder for feature extraction.
    
    Args:
        image_channels: Number of input channels
        image_size: Input image size
        latent_dim: Output latent dimension
        architecture: Architecture type ['resnet18', 'resnet34', 'custom']
    """
    
    def __init__(self, 
                 image_channels: int, 
                 image_size: int, 
                 latent_dim: int, 
                 architecture: str = 'resnet18'):
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
    
    def _build_resnet18_encoder(self):
        """Build ResNet18-based encoder."""
        # Load pretrained ResNet18
        resnet = models.resnet18(pretrained=False)
        
        # Modify first conv layer if needed (for grayscale or different input size)
        if self.image_channels != 3:
            resnet.conv1 = nn.Conv2d(self.image_channels, 64, kernel_size=7, 
                                    stride=2, padding=3, bias=False)
        
        # Remove final fully connected layer and average pooling
        modules = list(resnet.children())[:-2]  # Remove avgpool and fc
        self.features = nn.Sequential(*modules)
        
        # Calculate feature map size after convolutions
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.image_channels, self.image_size, self.image_size)
            feature_map = self.features(dummy_input)
            self.feature_map_size = feature_map.shape[1] * feature_map.shape[2] * feature_map.shape[3]
        
        # Add adaptive pooling and projection to latent_dim
        return nn.Sequential(
            self.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, self.latent_dim),
            nn.ReLU(inplace=True)
        )
    
    def _build_resnet34_encoder(self):
        """Build ResNet34-based encoder."""
        resnet = models.resnet34(pretrained=False)
        
        if self.image_channels != 3:
            resnet.conv1 = nn.Conv2d(self.image_channels, 64, kernel_size=7, 
                                    stride=2, padding=3, bias=False)
        
        modules = list(resnet.children())[:-2]
        self.features = nn.Sequential(*modules)
        
        return nn.Sequential(
            self.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, self.latent_dim),
            nn.ReLU(inplace=True)
        )
    
    def _build_custom_encoder(self):
        """Build custom CNN encoder for smaller images."""
        # Simple CNN for MNIST/CIFAR-10
        return nn.Sequential(
            # Block 1
            nn.Conv2d(self.image_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Global pooling and projection
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, self.latent_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images, shape (batch_size, channels, H, W)
        
        Returns:
            Latent representation, shape (batch_size, latent_dim)
        """
        return self.encoder(x)
