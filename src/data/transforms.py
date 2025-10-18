import torch
import torchvision.transforms as transforms
from typing import Dict, Any, List


def get_train_transforms(config: Dict[str, Any], image_size: int, image_channels: int) -> transforms.Compose:
    """
    Get training data transformations.
    
    Args:
        config: Augmentation configuration
        image_size: Target image size
        image_channels: Number of image channels (1 for grayscale, 3 for RGB)
    
    Returns:
        Composed transformations
    """
    transform_list = []
    
    # Random crop with padding
    if config.get('random_crop', True):
        padding = config.get('crop_padding', 4)
        transform_list.append(transforms.RandomCrop(image_size, padding=padding))
    
    # Random horizontal flip
    if config.get('random_horizontal_flip', True):
        flip_prob = config.get('flip_probability', 0.5)
        transform_list.append(transforms.RandomHorizontalFlip(p=flip_prob))
    
    # Color jitter (only for RGB images)
    if config.get('color_jitter', True) and image_channels == 3:
        jitter_params = config.get('jitter_params', {})
        transform_list.append(transforms.ColorJitter(
            brightness=jitter_params.get('brightness', 0.2),
            contrast=jitter_params.get('contrast', 0.2),
            saturation=jitter_params.get('saturation', 0.2),
            hue=jitter_params.get('hue', 0.1)
        ))
    
    # Convert to tensor
    transform_list.append(transforms.ToTensor())
    
    # Normalize
    if config.get('normalize', True):
        mean = config.get('normalization_mean', [0.5, 0.5, 0.5])
        std = config.get('normalization_std', [0.5, 0.5, 0.5])
        
        # Adjust for grayscale
        if image_channels == 1:
            mean = [mean[0]]
            std = [std[0]]
        
        transform_list.append(transforms.Normalize(mean=mean, std=std))
    
    return transforms.Compose(transform_list)


def get_test_transforms(config: Dict[str, Any], image_size: int, image_channels: int) -> transforms.Compose:
    """
    Get test data transformations.
    
    Args:
        config: Augmentation configuration
        image_size: Target image size
        image_channels: Number of image channels
    
    Returns:
        Composed transformations
    """
    transform_list = []
    
    # Resize if needed
    transform_list.append(transforms.Resize((image_size, image_size)))
    
    # Convert to tensor
    transform_list.append(transforms.ToTensor())
    
    # Normalize
    if config.get('normalize', True):
        mean = config.get('normalization_mean', [0.5, 0.5, 0.5])
        std = config.get('normalization_std', [0.5, 0.5, 0.5])
        
        # Adjust for grayscale
        if image_channels == 1:
            mean = [mean[0]]
            std = [std[0]]
        
        transform_list.append(transforms.Normalize(mean=mean, std=std))
    
    return transforms.Compose(transform_list)


def get_background_transforms(augmentation_strength: float = 0.8, 
                              image_size: int = 32,
                              image_channels: int = 3) -> transforms.Compose:
    """
    Get strong augmentation transforms for background class generation.
    
    Args:
        augmentation_strength: Strength of augmentation [0, 1]
        image_size: Target image size
        image_channels: Number of image channels
    
    Returns:
        Composed transformations
    """
    transform_list = []
    
    # Strong random crop
    padding = int(image_size * 0.25 * augmentation_strength)
    transform_list.append(transforms.RandomCrop(image_size, padding=padding))
    
    # Random horizontal flip
    transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
    
    # Random rotation
    degrees = int(45 * augmentation_strength)
    transform_list.append(transforms.RandomRotation(degrees=degrees))
    
    # Color jitter (only for RGB)
    if image_channels == 3:
        transform_list.append(transforms.ColorJitter(
            brightness=0.4 * augmentation_strength,
            contrast=0.4 * augmentation_strength,
            saturation=0.4 * augmentation_strength,
            hue=0.2 * augmentation_strength
        ))
    
    # Random perspective
    transform_list.append(transforms.RandomPerspective(
        distortion_scale=0.3 * augmentation_strength, 
        p=0.5
    ))
    
    # Convert to tensor
    transform_list.append(transforms.ToTensor())
    
    # Normalize
    mean = [0.5, 0.5, 0.5] if image_channels == 3 else [0.5]
    std = [0.5, 0.5, 0.5] if image_channels == 3 else [0.5]
    transform_list.append(transforms.Normalize(mean=mean, std=std))
    
    return transforms.Compose(transform_list)
