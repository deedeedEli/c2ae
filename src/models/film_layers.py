import torch
import torch.nn as nn


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation layer.
    
    Applies affine transformation to feature maps based on class embedding:
        FiLM(F, gamma, beta) = gamma * F + beta
    
    Args:
        num_features: Number of input feature channels
        embedding_dim: Dimension of class embedding
    """
    
    def __init__(self, num_features: int, embedding_dim: int):
        super(FiLM, self).__init__()
        self.num_features = num_features
        self.embedding_dim = embedding_dim
        
        # Generate gamma and beta from embeddings
        self.film_generator = FiLMGenerator(embedding_dim, num_features)
    
    def forward(self, features: torch.Tensor, class_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Feature maps, shape (batch_size, num_features, height, width)
            class_embedding: Class embeddings, shape (batch_size, embedding_dim)
        
        Returns:
            Modulated features, same shape as input features
        """
        gamma, beta = self.film_generator(class_embedding)
        
        # Apply FiLM transformation
        return gamma * features + beta


class FiLMGenerator(nn.Module):
    """
    Generates FiLM parameters (gamma, beta) from class embeddings.
    
    Args:
        embedding_dim: Dimension of class embedding
        num_features: Number of feature channels to modulate
        hidden_dim: Hidden layer dimension
    """
    
    def __init__(self, embedding_dim: int, num_features: int, hidden_dim: int = 256):
        super(FiLMGenerator, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_features = num_features
        
        # Network to generate gamma and beta
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * num_features)  # Output both gamma and beta
        )
    
    def forward(self, class_embedding: torch.Tensor):
        """
        Args:
            class_embedding: shape (batch_size, embedding_dim)
        
        Returns:
            tuple: (gamma, beta)
                - gamma: shape (batch_size, num_features, 1, 1)
                - beta: shape (batch_size, num_features, 1, 1)
        """
        # Generate parameters
        params = self.fc(class_embedding)  # (batch_size, 2 * num_features)
        
        # Split into gamma and beta
        gamma, beta = torch.chunk(params, 2, dim=1)  # Each: (batch_size, num_features)
        
        # Reshape for broadcasting with feature maps
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # (batch_size, num_features, 1, 1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)    # (batch_size, num_features, 1, 1)
        
        return gamma, beta
