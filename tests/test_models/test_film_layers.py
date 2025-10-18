import torch
import pytest
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.film_layers import FiLM, FiLMGenerator


def test_film_generator():
    """Test FiLM generator creates correct shapes."""
    embedding_dim = 128
    num_features = 64
    batch_size = 4
    
    generator = FiLMGenerator(embedding_dim, num_features)
    class_embedding = torch.randn(batch_size, embedding_dim)
    
    gamma, beta = generator(class_embedding)
    
    assert gamma.shape == (batch_size, num_features, 1, 1)
    assert beta.shape == (batch_size, num_features, 1, 1)


def test_film_layer():
    """Test FiLM layer modulates features correctly."""
    embedding_dim = 128
    num_features = 64
    batch_size = 4
    height, width = 8, 8
    
    film = FiLM(num_features, embedding_dim)
    features = torch.randn(batch_size, num_features, height, width)
    class_embedding = torch.randn(batch_size, embedding_dim)
    
    modulated = film(features, class_embedding)
    
    assert modulated.shape == features.shape
    assert not torch.allclose(modulated, features)  # Should be different


def test_film_deterministic():
    """Test FiLM produces same output for same input."""
    embedding_dim = 128
    num_features = 64
    batch_size = 4
    height, width = 8, 8
    
    film = FiLM(num_features, embedding_dim)
    film.eval()
    
    features = torch.randn(batch_size, num_features, height, width)
    class_embedding = torch.randn(batch_size, embedding_dim)
    
    output1 = film(features, class_embedding)
    output2 = film(features, class_embedding)
    
    assert torch.allclose(output1, output2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
