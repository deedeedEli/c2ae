# PROJECT_STRUCTURE.md - C2AE Architecture Blueprint

## Paper Summary

**C2AE (Class Conditioned Auto-Encoder) for Open-Set Recognition** addresses the challenge of recognizing known classes while detecting unknown/novel classes at test time. The method uses a class-conditioned autoencoder with Feature-wise Linear Modulation (FiLM) layers that reconstructs images based on class embeddings. During testing, large reconstruction errors indicate unknown classes since the model cannot properly reconstruct samples from unseen classes.

## Repository Tree

```
c2ae-reproduction/
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── c2ae.py                    # Main C2AE model class with encoder, decoder, FiLM layers
│   │   ├── encoder.py                 # Encoder network (ResNet-based feature extractor)
│   │   ├── decoder.py                 # Decoder network with FiLM conditioning
│   │   ├── film_layers.py             # FiLM (Feature-wise Linear Modulation) implementation
│   │   └── class_embedding.py         # Class embedding layer for conditioning
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset_loader.py          # Dataset loading for MNIST, SVHN, CIFAR10, TinyImageNet
│   │   ├── openset_split.py           # Open-set train/test split generation
│   │   ├── background_generator.py    # Background/outlier class generation
│   │   ├── transforms.py              # Data augmentation and preprocessing
│   │   └── data_utils.py              # Utility functions for data handling
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py                 # Main training loop with reconstruction loss
│   │   ├── losses.py                  # Custom loss functions (reconstruction + regularization)
│   │   └── scheduler.py               # Learning rate scheduling
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── openset_evaluator.py       # Open-set evaluation with AUROC, AUPR, CCR
│   │   ├── metrics.py                 # Implementation of all evaluation metrics
│   │   ├── threshold_selection.py     # Automatic threshold selection methods
│   │   └── visualization.py           # Result visualization (ROC curves, t-SNE, etc.)
│   ├── experiments/
│   │   ├── __init__.py
│   │   ├── run_experiment.py          # Main experiment execution script
│   │   ├── experiment_configs.py      # Experiment configuration definitions
│   │   └── logger.py                  # Experiment logging and tracking
│   └── utils/
│       ├── __init__.py
│       ├── config.py                  # Configuration management
│       ├── checkpoint.py              # Model checkpoint saving/loading
│       ├── reproducibility.py         # Random seed and reproducibility utilities
│       └── device_manager.py          # GPU/CPU device management
├── configs/
│   ├── default.yaml                   # Default hyperparameters
│   ├── mnist.yaml                     # MNIST-specific configuration
│   ├── svhn.yaml                      # SVHN-specific configuration
│   ├── cifar10.yaml                   # CIFAR10-specific configuration
│   ├── cifar_plus.yaml                # CIFAR+10/+50 configuration
│   └── tiny_imagenet.yaml             # Tiny ImageNet configuration
├── tests/
│   ├── __init__.py
│   ├── test_models/
│   │   ├── test_c2ae.py               # C2AE model unit tests
│   │   ├── test_film_layers.py        # FiLM layer functionality tests
│   │   └── test_encoder_decoder.py    # Encoder/decoder architecture tests
│   ├── test_data/
│   │   ├── test_dataset_loader.py     # Dataset loading tests
│   │   ├── test_openset_split.py      # Open-set splitting tests
│   │   └── test_background_gen.py     # Background generation tests
│   ├── test_training/
│   │   ├── test_trainer.py            # Training loop tests
│   │   └── test_losses.py             # Loss function tests
│   └── test_evaluation/
│       ├── test_metrics.py            # Metric computation tests
│       └── test_evaluator.py          # Evaluator functionality tests
├── scripts/
│   ├── download_datasets.sh           # Dataset download automation
│   ├── setup_environment.sh           # Environment setup script
│   └── run_all_experiments.sh         # Run complete experimental suite
├── data/
│   ├── raw/                           # Raw downloaded datasets
│   ├── processed/                     # Preprocessed data
│   └── openset_splits/                # Open-set train/test splits
├── results/
│   ├── checkpoints/                   # Model checkpoints
│   ├── logs/                          # Training and evaluation logs
│   ├── metrics/                       # Saved metric results
│   └── visualizations/                # Generated plots and figures
├── notebooks/
│   ├── 01_data_exploration.ipynb      # Dataset analysis
│   ├── 02_model_analysis.ipynb        # Model architecture visualization
│   └── 03_results_analysis.ipynb      # Results comparison and visualization
├── requirements.txt                    # Python dependencies with versions
├── setup.py                           # Package installation configuration
├── README.md                          # Project overview and usage guide
└── .gitignore                         # Git ignore patterns

```

## Module Interface Map

### Core Model Components

#### 1. `src/models/c2ae.py`

```python
class C2AE(nn.Module):
    """
    Main Class Conditioned Auto-Encoder model.
    
    Args:
        num_classes (int): Number of known classes (+1 for background)
        embedding_dim (int): Dimension of class embedding (default: 128)
        encoder_arch (str): Encoder architecture ['resnet18', 'resnet34', 'custom']
        latent_dim (int): Dimension of latent representation (default: 512)
        image_channels (int): Number of input image channels (1 for grayscale, 3 for RGB)
        image_size (int): Input image size (assumes square images)
    """
    
    def __init__(self, num_classes, embedding_dim=128, encoder_arch='resnet18', 
                 latent_dim=512, image_channels=3, image_size=32)
    
    def forward(self, x, class_labels):
        """
        Forward pass through C2AE.
        
        Args:
            x (torch.Tensor): Input images, shape (batch_size, channels, height, width)
            class_labels (torch.Tensor): Class labels for conditioning, shape (batch_size,)
        
        Returns:
            dict: {
                'reconstructed': torch.Tensor, shape (batch_size, channels, height, width)
                'latent': torch.Tensor, shape (batch_size, latent_dim)
                'class_embedding': torch.Tensor, shape (batch_size, embedding_dim)
            }
        """
    
    def compute_reconstruction_error(self, x, reconstructed, reduction='none'):
        """
        Compute pixel-wise reconstruction error.
        
        Args:
            x (torch.Tensor): Original images
            reconstructed (torch.Tensor): Reconstructed images
            reduction (str): 'none', 'mean', 'sum'
        
        Returns:
            torch.Tensor: Reconstruction errors
                - If reduction='none': shape (batch_size, channels, height, width)
                - If reduction='mean' or 'sum': shape (batch_size,)
        """
    
    def predict_openset(self, x, known_classes, threshold=None, return_scores=True):
        """
        Predict whether samples are known or unknown classes.
        
        Args:
            x (torch.Tensor): Input images
            known_classes (list[int]): List of known class indices
            threshold (float): Decision threshold for unknown detection
            return_scores (bool): Whether to return reconstruction scores
        
        Returns:
            dict: {
                'predictions': torch.Tensor, shape (batch_size,) - predicted class (-1 for unknown)
                'scores': torch.Tensor, shape (batch_size, num_known_classes) - reconstruction errors per class
                'is_known': torch.Tensor, shape (batch_size,) - boolean mask
            }
        """
```

#### 2. `src/models/film_layers.py`

```python
class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation layer.
    
    Applies affine transformation to feature maps based on class embedding:
        FiLM(F, gamma, beta) = gamma * F + beta
    
    Args:
        num_features (int): Number of input feature channels
        embedding_dim (int): Dimension of class embedding
    """
    
    def __init__(self, num_features, embedding_dim)
    
    def forward(self, features, class_embedding):
        """
        Args:
            features (torch.Tensor): Feature maps, shape (batch_size, num_features, height, width)
            class_embedding (torch.Tensor): Class embeddings, shape (batch_size, embedding_dim)
        
        Returns:
            torch.Tensor: Modulated features, same shape as input features
        """


class FiLMGenerator(nn.Module):
    """
    Generates FiLM parameters (gamma, beta) from class embeddings.
    
    Args:
        embedding_dim (int): Dimension of class embedding
        num_features (int): Number of feature channels to modulate
        hidden_dim (int): Hidden layer dimension (default: 256)
    """
    
    def __init__(self, embedding_dim, num_features, hidden_dim=256)
    
    def forward(self, class_embedding):
        """
        Args:
            class_embedding (torch.Tensor): shape (batch_size, embedding_dim)
        
        Returns:
            tuple: (gamma, beta)
                - gamma: shape (batch_size, num_features, 1, 1)
                - beta: shape (batch_size, num_features, 1, 1)
        """
```

#### 3. `src/models/encoder.py`

```python
class Encoder(nn.Module):
    """
    CNN-based encoder for feature extraction.
    
    Args:
        image_channels (int): Number of input channels
        image_size (int): Input image size
        latent_dim (int): Output latent dimension
        architecture (str): Architecture type ['resnet18', 'resnet34', 'custom']
    """
    
    def __init__(self, image_channels, image_size, latent_dim, architecture='resnet18')
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input images, shape (batch_size, channels, H, W)
        
        Returns:
            torch.Tensor: Latent representation, shape (batch_size, latent_dim)
        """
```

#### 4. `src/models/decoder.py`

```python
class ConditionalDecoder(nn.Module):
    """
    Decoder with FiLM conditioning at multiple layers.
    
    Args:
        latent_dim (int): Dimension of latent input
        embedding_dim (int): Dimension of class embedding
        image_channels (int): Number of output image channels
        image_size (int): Output image size
        film_positions (list[int]): Layer indices where FiLM is applied
    """
    
    def __init__(self, latent_dim, embedding_dim, image_channels, image_size, 
                 film_positions=[0, 1, 2, 3])
    
    def forward(self, latent, class_embedding):
        """
        Args:
            latent (torch.Tensor): Latent representation, shape (batch_size, latent_dim)
            class_embedding (torch.Tensor): Class embeddings, shape (batch_size, embedding_dim)
        
        Returns:
            torch.Tensor: Reconstructed images, shape (batch_size, channels, H, W)
        """
```

### Data Pipeline Components

#### 5. `src/data/dataset_loader.py`

```python
class OpenSetDataset(Dataset):
    """
    PyTorch Dataset for open-set recognition.
    
    Args:
        dataset_name (str): 'mnist', 'svhn', 'cifar10', 'cifar+10', 'cifar+50', 'tiny_imagenet'
        known_classes (list[int]): List of known class indices
        split (str): 'train', 'test_known', 'test_unknown'
        include_background (bool): Whether to include background class in training
        transform (callable): Data transformation pipeline
        data_root (str): Root directory for datasets
    """
    
    def __init__(self, dataset_name, known_classes, split='train', 
                 include_background=False, transform=None, data_root='./data')
    
    def __getitem__(self, idx):
        """
        Returns:
            tuple: (image, label, is_known)
                - image: torch.Tensor, shape (C, H, W)
                - label: int, original class label
                - is_known: bool, whether sample is from known classes
        """


def get_openset_loaders(dataset_name, known_classes, batch_size, 
                       num_workers=4, data_root='./data'):
    """
    Create train and test data loaders for open-set experiments.
    
    Args:
        dataset_name (str): Dataset name
        known_classes (list[int]): Known class indices
        batch_size (int): Batch size
        num_workers (int): Number of data loading workers
        data_root (str): Data directory
    
    Returns:
        dict: {
            'train': DataLoader for training (known + background)
            'test_known': DataLoader for known class testing
            'test_unknown': DataLoader for unknown class testing
        }
    """
```

#### 6. `src/data/background_generator.py`

```python
class BackgroundGenerator:
    """
    Generates background/outlier samples for training.
    
    Uses random crops, rotations, color jittering to create OOD samples
    from the same dataset or external sources.
    
    Args:
        source_dataset (str): Dataset to sample from
        known_classes (list[int]): Known classes to exclude
        augmentation_strength (float): Strength of augmentation [0, 1]
    """
    
    def __init__(self, source_dataset, known_classes, augmentation_strength=0.8)
    
    def generate_background_batch(self, batch_size):
        """
        Generate batch of background samples.
        
        Args:
            batch_size (int): Number of samples to generate
        
        Returns:
            torch.Tensor: Background images, shape (batch_size, C, H, W)
        """
```

### Training Components

#### 7. `src/training/trainer.py`

```python
class C2AETrainer:
    """
    Trainer for C2AE model.
    
    Args:
        model (C2AE): C2AE model instance
        optimizer (torch.optim.Optimizer): Optimizer
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler
        device (torch.device): Training device
        config (dict): Training configuration
    """
    
    def __init__(self, model, optimizer, scheduler, device, config)
    
    def train_epoch(self, train_loader, epoch):
        """
        Train for one epoch.
        
        Args:
            train_loader (DataLoader): Training data loader
            epoch (int): Current epoch number
        
        Returns:
            dict: {
                'loss': float, average training loss
                'reconstruction_error': float, average reconstruction error
                'learning_rate': float, current learning rate
            }
        """
    
    def validate(self, val_loader):
        """
        Validation step.
        
        Args:
            val_loader (DataLoader): Validation data loader
        
        Returns:
            dict: Validation metrics
        """
    
    def fit(self, train_loader, val_loader, num_epochs, checkpoint_dir):
        """
        Full training loop.
        
        Args:
            train_loader (DataLoader): Training data
            val_loader (DataLoader): Validation data
            num_epochs (int): Number of training epochs
            checkpoint_dir (str): Directory to save checkpoints
        
        Returns:
            dict: Training history
        """
```

### Evaluation Components

#### 8. `src/evaluation/openset_evaluator.py`

```python
class OpenSetEvaluator:
    """
    Evaluator for open-set recognition performance.
    
    Args:
        model (C2AE): Trained C2AE model
        known_classes (list[int]): List of known class indices
        device (torch.device): Evaluation device
    """
    
    def __init__(self, model, known_classes, device)
    
    def evaluate(self, test_known_loader, test_unknown_loader):
        """
        Comprehensive open-set evaluation.
        
        Args:
            test_known_loader (DataLoader): Known class test data
            test_unknown_loader (DataLoader): Unknown class test data
        
        Returns:
            dict: {
                'auroc': float, Area Under ROC Curve
                'aupr': float, Area Under Precision-Recall Curve
                'fpr_at_95_tpr': float, False Positive Rate at 95% TPR
                'ccr_at_fpr_5': float, Correct Classification Rate at 5% FPR
                'f1_score': float, F1 score at optimal threshold
                'optimal_threshold': float, Automatically selected threshold
                'per_class_accuracy': dict, Accuracy for each known class
            }
        """
    
    def compute_openset_scores(self, data_loader):
        """
        Compute reconstruction-based openness scores.
        
        Args:
            data_loader (DataLoader): Data to score
        
        Returns:
            tuple: (scores, labels, predictions)
                - scores: np.ndarray, shape (num_samples,)
                - labels: np.ndarray, shape (num_samples,)
                - predictions: np.ndarray, shape (num_samples,)
        """
```

#### 9. `src/evaluation/metrics.py`

```python
def compute_auroc(known_scores, unknown_scores):
    """
    Compute Area Under ROC Curve.
    
    Args:
        known_scores (np.ndarray): Reconstruction errors for known classes
        unknown_scores (np.ndarray): Reconstruction errors for unknown classes
    
    Returns:
        float: AUROC value in [0, 1]
    """


def compute_aupr(known_scores, unknown_scores):
    """
    Compute Area Under Precision-Recall Curve.
    
    Args:
        known_scores (np.ndarray): Scores for known classes
        unknown_scores (np.ndarray): Scores for unknown classes
    
    Returns:
        float: AUPR value in [0, 1]
    """


def compute_ccr_at_fpr(known_scores, unknown_scores, fpr_threshold=0.05):
    """
    Compute Correct Classification Rate at specific False Positive Rate.
    
    Args:
        known_scores (np.ndarray): Scores for known classes
        unknown_scores (np.ndarray): Scores for unknown classes
        fpr_threshold (float): Desired FPR threshold
    
    Returns:
        tuple: (ccr, threshold)
            - ccr: float, CCR value at specified FPR
            - threshold: float, score threshold for this FPR
    """
```

## Technology Stack

### Core Dependencies

```
Python: 3.8.x - 3.10.x
PyTorch: 2.0.0+cu118 (CUDA 11.8 for GPU support)
torchvision: 0.15.0
numpy: 1.24.0
scikit-learn: 1.2.0 (for metric computation)
scipy: 1.10.0 (for statistical functions)
matplotlib: 3.7.0 (for visualization)
seaborn: 0.12.0 (for advanced plotting)
tensorboard: 2.12.0 (for training visualization)
PyYAML: 6.0 (for configuration files)
tqdm: 4.65.0 (for progress bars)
Pillow: 9.5.0 (for image processing)
```

### Development Dependencies

```
pytest: 7.3.0 (for unit testing)
pytest-cov: 4.0.0 (for coverage reports)
black: 23.3.0 (code formatting)
flake8: 6.0.0 (linting)
mypy: 1.2.0 (type checking)
jupyter: 1.0.0 (for notebooks)
```

### Hardware Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 16 GB
- Storage: 50 GB
- GPU: Not required but strongly recommended

**Recommended:**
- CPU: 8+ cores
- RAM: 32 GB
- Storage: 100 GB
- GPU: NVIDIA GPU with 8+ GB VRAM (e.g., RTX 2080, V100)
- CUDA: 11.8+
- cuDNN: 8.6+

**Training Time Estimates:**
- MNIST: ~10 minutes (CPU), ~2 minutes (GPU)
- SVHN: ~30 minutes (CPU), ~5 minutes (GPU)
- CIFAR-10: ~1 hour (CPU), ~10 minutes (GPU)
- Tiny ImageNet: ~8 hours (CPU), ~1 hour (GPU)

## Configuration Schema

### Default Configuration (`configs/default.yaml`)

```yaml
# Model Architecture
model:
  encoder_arch: "resnet18"           # Encoder architecture
  embedding_dim: 128                 # Class embedding dimension
  latent_dim: 512                    # Latent space dimension
  film_positions: [0, 1, 2, 3]      # Layers where FiLM is applied

# Training Hyperparameters
training:
  num_epochs: 100
  batch_size: 128
  learning_rate: 0.001
  weight_decay: 0.0001
  optimizer: "adam"                  # Options: adam, sgd, adamw
  scheduler: "step"                  # Options: step, cosine, plateau
  scheduler_params:
    step_size: 30
    gamma: 0.1
  
  # Loss weights
  reconstruction_weight: 1.0
  
  # Early stopping
  early_stopping_patience: 20
  early_stopping_metric: "val_loss"

# Data Configuration
data:
  dataset: "cifar10"                 # mnist, svhn, cifar10, cifar+10, cifar+50, tiny_imagenet
  data_root: "./data"
  num_workers: 4
  pin_memory: true
  
  # Open-set configuration
  num_known_classes: 6               # Number of known classes during training
  known_class_indices: [0, 1, 2, 3, 4, 5]  # Specific known classes (null for random)
  
  # Background class
  use_background: true
  background_ratio: 0.3              # Ratio of background samples in training
  background_augmentation: 0.8       # Augmentation strength for background

# Data Augmentation
augmentation:
  train:
    random_crop: true
    crop_padding: 4
    random_horizontal_flip: true
    flip_probability: 0.5
    color_jitter: true
    jitter_params:
      brightness: 0.2
      contrast: 0.2
      saturation: 0.2
      hue: 0.1
    random_rotation: false
    normalize: true
    normalization_mean: [0.5, 0.5, 0.5]
    normalization_std: [0.5, 0.5, 0.5]
  
  test:
    normalize: true
    normalization_mean: [0.5, 0.5, 0.5]
    normalization_std: [0.5, 0.5, 0.5]

# Evaluation Configuration
evaluation:
  metrics: ["auroc", "aupr", "ccr_at_fpr", "f1_score"]
  fpr_thresholds: [0.01, 0.05, 0.1]
  threshold_selection: "optimal_f1"  # Options: optimal_f1, fixed, percentile
  
  # Open-set scoring
  score_aggregation: "min"           # How to aggregate multi-class scores: min, mean, max

# Experiment Configuration
experiment:
  name: "c2ae_default"
  seed: 42
  log_interval: 10                   # Log every N batches
  checkpoint_interval: 10            # Save checkpoint every N epochs
  save_best_only: true
  
  # Logging
  use_tensorboard: true
  tensorboard_dir: "./results/tensorboard"
  save_dir: "./results"

# Hardware Configuration
hardware:
  device: "cuda"                     # cuda, cpu, mps (for Apple Silicon)
  cuda_deterministic: false          # Set to true for reproducibility (slower)
  num_gpus: 1
```

## Entry Points

### 1. Main Training Script

```bash
# Train C2AE on CIFAR-10 with default settings
python -m src.experiments.run_experiment \
    --config configs/cifar10.yaml \
    --mode train

# Train with custom parameters
python -m src.experiments.run_experiment \
    --config configs/cifar10.yaml \
    --mode train \
    --batch_size 256 \
    --learning_rate 0.0005 \
    --num_epochs 150 \
    --known_classes 0,1,2,3,4,5
```

### 2. Evaluation Script

```bash
# Evaluate trained model
python -m src.experiments.run_experiment \
    --config configs/cifar10.yaml \
    --mode evaluate \
    --checkpoint results/checkpoints/best_model.pth

# Evaluate with different threshold
python -m src.experiments.run_experiment \
    --config configs/cifar10.yaml \
    --mode evaluate \
    --checkpoint results/checkpoints/best_model.pth \
    --threshold 0.15
```

### 3. Full Experimental Suite

```bash
# Run all experiments from paper
bash scripts/run_all_experiments.sh

# Run specific dataset experiments
python -m src.experiments.run_experiment \
    --config configs/mnist.yaml \
    --mode full \
    --trials 5  # Run 5 trials with different seeds
```

### 4. API Usage Example

```python
from src.models.c2ae import C2AE
from src.data.dataset_loader import get_openset_loaders
from src.training.trainer import C2AETrainer
from src.evaluation.openset_evaluator import OpenSetEvaluator

# Initialize model
model = C2AE(
    num_classes=7,  # 6 known + 1 background
    embedding_dim=128,
    encoder_arch='resnet18',
    latent_dim=512,
    image_channels=3,
    image_size=32
)

# Load data
loaders = get_openset_loaders(
    dataset_name='cifar10',
    known_classes=[0, 1, 2, 3, 4, 5],
    batch_size=128
)

# Train
trainer = C2AETrainer(model, optimizer, scheduler, device, config)
history = trainer.fit(
    train_loader=loaders['train'],
    val_loader=loaders['test_known'],
    num_epochs=100,
    checkpoint_dir='./checkpoints'
)

# Evaluate
evaluator = OpenSetEvaluator(model, known_classes=[0,1,2,3,4,5], device=device)
results = evaluator.evaluate(
    test_known_loader=loaders['test_known'],
    test_unknown_loader=loaders['test_unknown']
)

print(f"AUROC: {results['auroc']:.4f}")
print(f"AUPR: {results['aupr']:.4f}")
print(f"CCR@FPR=5%: {results['ccr_at_fpr_5']:.4f}")
```

## Cross-File Dependencies

```
c2ae.py
  ├─> encoder.py (Encoder class)
  ├─> decoder.py (ConditionalDecoder class)
  ├─> film_layers.py (FiLM, FiLMGenerator classes)
  └─> class_embedding.py (ClassEmbedding class)

trainer.py
  ├─> c2ae.py (C2AE model)
  ├─> losses.py (loss functions)
  ├─> scheduler.py (LR scheduling)
  └─> checkpoint.py (model saving)

run_experiment.py
  ├─> dataset_loader.py (data loading)
  ├─> c2ae.py (model)
  ├─> trainer.py (training)
  ├─> openset_evaluator.py (evaluation)
  └─> config.py (configuration management)

openset_evaluator.py
  ├─> c2ae.py (model inference)
  ├─> metrics.py (metric computation)
  └─> threshold_selection.py (threshold selection)
```

## Data Flow Architecture

```
Raw Dataset
    ↓
dataset_loader.py: Load & split into known/unknown
    ↓
transforms.py: Apply augmentation
    ↓
background_generator.py: Add background class samples
    ↓
DataLoader: Batch creation
    ↓
C2AE Model:
    ├─> Encoder: x → latent
    ├─> ClassEmbedding: class_label → class_embedding
    └─> Decoder: (latent, class_embedding) → reconstructed_x
    ↓
Loss Computation: MSE(x, reconstructed_x)
    ↓
Optimizer: Update model weights
    ↓
Evaluation:
    ├─> Compute reconstruction errors for all class hypotheses
    ├─> Select minimum error as openness score
    ├─> Compare scores for known vs unknown classes
    └─> Compute AUROC, AUPR, CCR metrics
```

## Testing Strategy

Each module includes comprehensive tests:

1. **Unit Tests**: Test individual functions and classes in isolation
2. **Integration Tests**: Test component interactions
3. **Regression Tests**: Ensure reproducibility of results
4. **Performance Tests**: Verify training speed and memory usage

**Run all tests:**
```bash
pytest tests/ -v --cov=src --cov-report=html
```

**Run specific test suite:**
```bash
pytest tests/test_models/ -v
pytest tests/test_data/ -v
pytest tests/test_evaluation/ -v
```

## Development Workflow

1. **Setup**: Run `bash scripts/setup_environment.sh`
2. **Data Preparation**: Run `bash scripts/download_datasets.sh`
3. **Implementation**: Follow phases in IMPLEMENTATION_PLAN.md
4. **Testing**: Write and run tests after each module
5. **Training**: Execute training with configurations
6. **Evaluation**: Run evaluation scripts and compare with paper results
7. **Iteration**: Adjust hyperparameters if results don't match

## Success Criteria

✅ Model architecture matches paper description
✅ Training converges without numerical issues
✅ Reconstruction quality is visually acceptable
✅ Open-set metrics match paper results (±2% tolerance):
   - CIFAR-10: AUROC > 0.90, AUPR > 0.88
   - MNIST: AUROC > 0.98, AUPR > 0.97
   - SVHN: AUROC > 0.88, AUPR > 0.85
✅ All unit tests pass
✅ Code follows style guidelines
✅ Documentation is complete
