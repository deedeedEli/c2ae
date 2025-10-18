import torch
import torch.nn as nn


class ClassEmbedding(nn.Module):
    """
    Learnable class embedding layer.
    
    Args:
        num_classes: Number of classes (including background class)
        embedding_dim: Dimension of embedding vectors
    """
    
    def __init__(self, num_classes: int, embedding_dim: int):
        super(ClassEmbedding, self).__init__()
        
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        
        # Learnable embedding layer
        self.embedding = nn.Embedding(num_classes, embedding_dim)
        
        # Initialize embeddings
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)
    
    def forward(self, class_labels: torch.Tensor) -> torch.Tensor:
        """
        Get embeddings for given class labels.
        
        Args:
            class_labels: Class label indices, shape (batch_size,)
        
        Returns:
            Class embeddings, shape (batch_size, embedding_dim)
        """
        return self.embedding(class_labels)
