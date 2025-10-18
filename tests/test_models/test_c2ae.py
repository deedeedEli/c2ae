import torch
import pytest
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.c2ae import C2AE


def test_c2ae_forward():
    """Test C2AE forward pass."""
    batch_size = 4
    num_classes = 7
    image_size = 32
    image_channels = 3
    
    model = C2AE(
        num_classes=num_classes,
        embedding_dim=128,
        encoder_arch='custom',
        latent_dim=256,
        image_channels=image_channels,
        image_size=image_size
    )
    
    images = torch.randn(batch_size, image_channels, image_size, image_size)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    outputs = model(images, labels)
    
    assert 'reconstructed' in outputs
    assert 'latent' in outputs
    assert 'class_embedding' in outputs
    assert outputs['reconstructed'].shape == images.shape
    assert outputs['latent'].shape == (batch_size, 256)
    assert outputs['class_embedding'].shape == (batch_size, 128)


def test_c2ae_reconstruction_error():
    """Test reconstruction error computation."""
    batch_size = 4
    image_size = 32
    image_channels = 3
    
    model = C2AE(
        num_classes=7,
        image_channels=image_channels,
        image_size=image_size,
        encoder_arch='custom'
    )
    
    images = torch.randn(batch_size, image_channels, image_size, image_size)
    reconstructed = torch.randn(batch_size, image_channels, image_size, image_size)
    
    # Test different reductions
    error_none = model.compute_reconstruction_error(images, reconstructed, reduction='none')
    error_mean = model.compute_reconstruction_error(images, reconstructed, reduction='mean')
    error_sum = model.compute_reconstruction_error(images, reconstructed, reduction='sum')
    
    assert error_none.shape == images.shape
    assert error_mean.shape == (batch_size,)
    assert error_sum.shape == (batch_size,)


def test_c2ae_predict_openset():
    """Test open-set prediction."""
    batch_size = 4
    num_classes = 7
    known_classes = [0, 1, 2, 3, 4, 5]
    image_size = 32
    image_channels = 3
    
    model = C2AE(
        num_classes=num_classes,
        image_channels=image_channels,
        image_size=image_size,
        encoder_arch='custom'
    )
    model.eval()
    
    images = torch.randn(batch_size, image_channels, image_size, image_size)
    
    results = model.predict_openset(images, known_classes, threshold=0.5)
    
    assert 'predictions' in results
    assert 'is_known' in results
    assert 'min_reconstruction_error' in results
    assert results['predictions'].shape == (batch_size,)
    assert results['is_known'].shape == (batch_size,)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
