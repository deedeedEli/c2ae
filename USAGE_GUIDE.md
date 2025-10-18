# C2AE Usage Guide

Complete guide for using the C2AE (Class Conditioned Auto-Encoder) reproduction project.

---

## Table of Contents
1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Training](#training)
4. [Evaluation](#evaluation)
5. [Configuration](#configuration)
6. [Python API](#python-api)
7. [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites
- Python 3.8-3.10
- NVIDIA GPU with CUDA 11.8+ (recommended, but CPU works too)
- 16GB RAM minimum
- 50GB storage for datasets and results

### Setup Steps

1. **Clone the repository** (if not already done):
```bash
cd /path/to/c2ae-reproduction
```

2. **Run setup script**:
```bash
bash scripts/setup_environment.sh
```

3. **Activate virtual environment**:
```bash
source venv/bin/activate
```

4. **Verify installation**:
```bash
python scripts/verify_installation.py
```

5. **Download datasets**:
```bash
bash scripts/download_datasets.sh
```

---

## Quick Start

### 5-Minute Demo

Run the quick start example (trains for only 5 epochs):

```bash
python examples/quick_start.py
```

This will:
- Load MNIST dataset
- Create a C2AE model
- Train for 5 epochs
- Evaluate open-set performance

**Expected output** (after 5 epochs):
```
AUROC: ~0.85-0.90
AUPR: ~0.80-0.85
Known Class Accuracy: ~0.90-0.95
```

---

## Training

### Basic Training

Train on MNIST:
```bash
python -m src.experiments.run_experiment \
    --config configs/mnist.yaml \
    --mode train
```

Train on CIFAR-10:
```bash
python -m src.experiments.run_experiment \
    --config configs/cifar10.yaml \
    --mode train
```

### Advanced Training Options

**Override configuration parameters**:
```bash
python -m src.experiments.run_experiment \
    --config configs/cifar10.yaml \
    --mode train \
    --batch_size 256 \
    --learning_rate 0.0005 \
    --num_epochs 150
```

**Train with different seed**:
```bash
python -m src.experiments.run_experiment \
    --config configs/mnist.yaml \
    --mode train \
    --seed 123
```

### Training Outputs

Training creates the following outputs:

```
results/
├── checkpoints/
│   ├── checkpoint_epoch_10.pth
│   ├── checkpoint_epoch_20.pth
│   ├── ...
│   └── best_model.pth
├── logs/
│   └── c2ae_mnist_20241018_143022/
│       ├── hyperparameters.json
│       └── metrics_history.json
└── tensorboard/
    └── c2ae_mnist_20241018_143022/
        └── events.out.tfevents...
```

### Monitor Training

Start TensorBoard:
```bash
tensorboard --logdir results/tensorboard --port 6006
```

Then open browser to: http://localhost:6006

---

## Evaluation

### Evaluate Trained Model

Evaluate best checkpoint:
```bash
python -m src.experiments.run_experiment \
    --config configs/cifar10.yaml \
    --mode evaluate \
    --checkpoint results/checkpoints/best_model.pth
```

### Evaluation Metrics

The evaluation outputs:
- **AUROC**: Area Under ROC Curve (higher is better)
- **AUPR**: Area Under Precision-Recall Curve (higher is better)
- **FPR at 95% TPR**: False Positive Rate at 95% True Positive Rate (lower is better)
- **CCR at FPR**: Correct Classification Rate at various FPR thresholds
- **Known Class Accuracy**: Classification accuracy on known classes

### Full Pipeline (Train + Evaluate)

Run complete pipeline:
```bash
python -m src.experiments.run_experiment \
    --config configs/mnist.yaml \
    --mode full
```

This will:
1. Train the model
2. Save checkpoints
3. Load best model
4. Evaluate on test sets
5. Save results to `results/metrics/evaluation_results.json`

---

## Configuration

### Configuration Files

Configuration files are located in `configs/`:
- `default.yaml`: Default configuration
- `mnist.yaml`: MNIST-specific settings
- `cifar10.yaml`: CIFAR-10-specific settings

### Configuration Structure

```yaml
model:
  encoder_arch: "resnet18"       # Encoder: resnet18, resnet34, custom
  embedding_dim: 128             # Class embedding dimension
  latent_dim: 512               # Latent space dimension
  film_positions: [0, 1, 2, 3]  # Where to apply FiLM layers

training:
  num_epochs: 100               # Number of training epochs
  batch_size: 128              # Batch size
  learning_rate: 0.001         # Learning rate
  weight_decay: 0.0001         # Weight decay for regularization
  early_stopping_patience: 20  # Early stopping patience

data:
  dataset: "cifar10"           # Dataset: mnist, cifar10, svhn
  num_known_classes: 6         # Number of known classes
  known_class_indices: [0,1,2,3,4,5]  # Which classes are known
  background_ratio: 0.3        # Background class ratio

experiment:
  name: "c2ae_experiment"      # Experiment name
  seed: 42                     # Random seed
  save_dir: "./results"        # Results directory
```

### Creating Custom Configurations

1. Copy an existing config:
```bash
cp configs/cifar10.yaml configs/my_experiment.yaml
```

2. Edit parameters as needed

3. Run with your config:
```bash
python -m src.experiments.run_experiment \
    --config configs/my_experiment.yaml \
    --mode train
```

---

## Python API

### Basic Usage

```python
import torch
from src.models.c2ae import C2AE
from src.data.dataset_loader import get_openset_loaders
from src.utils.config import Config
from src.utils.reproducibility import set_seed

# Setup
config = Config('configs/cifar10.yaml')
set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create model
model = C2AE(
    num_classes=7,
    embedding_dim=128,
    encoder_arch='resnet18',
    latent_dim=512,
    image_channels=3,
    image_size=32
).to(device)

# Load data
loaders = get_openset_loaders(
    dataset_name='cifar10',
    known_classes=[0, 1, 2, 3, 4, 5],
    batch_size=128,
    config=config.config
)

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# ... training loop ...

# Inference
model.eval()
with torch.no_grad():
    outputs = model(images, labels)
    reconstructed = outputs['reconstructed']
```

### Open-Set Prediction

```python
from src.evaluation.openset_evaluator import OpenSetEvaluator

# Create evaluator
evaluator = OpenSetEvaluator(
    model=model,
    known_classes=[0, 1, 2, 3, 4, 5],
    device=device
)

# Evaluate
results = evaluator.evaluate(
    test_known_loader=loaders['test_known'],
    test_unknown_loader=loaders['test_unknown']
)

print(f"AUROC: {results['auroc']:.4f}")
print(f"AUPR: {results['aupr']:.4f}")
```

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
**Error**: `RuntimeError: CUDA out of memory`

**Solution**: Reduce batch size
```bash
python -m src.experiments.run_experiment \
    --config configs/cifar10.yaml \
    --batch_size 64  # or lower
```

#### 2. Training Loss Not Decreasing
**Possible causes**:
- Learning rate too high/low
- Model architecture mismatch
- Data preprocessing issues

**Solutions**:
```bash
# Try lower learning rate
python -m src.experiments.run_experiment \
    --config configs/cifar10.yaml \
    --learning_rate 0.0005

# Check data loading
python -c "from src.data.dataset_loader import get_openset_loaders; \
           loaders = get_openset_loaders('cifar10', [0,1,2,3,4,5], 32, {}); \
           print('Data loaded successfully')"
```

#### 3. Poor Open-Set Performance
**Solutions**:
- Train for more epochs (100+)
- Check that background class is included
- Verify data normalization
- Ensure sufficient training data

#### 4. Dataset Download Fails
**Manual download**:
```python
from torchvision import datasets
datasets.CIFAR10(root='./data/raw', train=True, download=True)
datasets.CIFAR10(root='./data/raw', train=False, download=True)
```

#### 5. Import Errors
**Solution**: Verify installation
```bash
python scripts/verify_installation.py
```

If dependencies are missing:
```bash
pip install -r requirements.txt
```

---

## Performance Tips

### Speed Up Training

1. **Use GPU**:
```yaml
# In config file
hardware:
  device: "cuda"
```

2. **Increase batch size** (if GPU memory allows):
```bash
--batch_size 256
```

3. **Reduce data loading time**:
```yaml
data:
  num_workers: 8  # Increase workers
  pin_memory: true
```

### Improve Results

1. **Train longer**:
```bash
--num_epochs 150
```

2. **Use data augmentation**:
```yaml
augmentation:
  train:
    random_crop: true
    random_horizontal_flip: true
    color_jitter: true
```

3. **Try different architectures**:
```yaml
model:
  encoder_arch: "resnet34"  # Larger model
```

4. **Tune hyperparameters**:
```bash
--learning_rate 0.0005
--batch_size 256
```

---

## Running Tests

Run all tests:
```bash
pytest tests/ -v
```

Run specific test:
```bash
pytest tests/test_models/test_c2ae.py -v
```

Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html
```

---

## Expected Results

### MNIST (6 known, 4 unknown)
- Training Time: ~2-3 hours (100 epochs, GPU)
- Expected AUROC: 0.985 ± 0.02
- Expected AUPR: 0.975 ± 0.02
- Known Class Accuracy: ~0.99

### CIFAR-10 (6 known, 4 unknown)
- Training Time: ~16-20 hours (100 epochs, GPU)
- Expected AUROC: 0.912 ± 0.02
- Expected AUPR: 0.896 ± 0.02
- Known Class Accuracy: ~0.90

### SVHN (6 known, 4 unknown)
- Training Time: ~8-10 hours (100 epochs, GPU)
- Expected AUROC: 0.878 ± 0.03
- Expected AUPR: 0.856 ± 0.03
- Known Class Accuracy: ~0.95

---

## Additional Resources

- **Paper**: [C2AE CVPR 2019 Paper](Oza_C2AE_Class_Conditioned_Auto-Encoder_for_Open-Set_Recognition_CVPR_2019_paper.pdf)
- **Documentation**: See `DOCUMENTATION_INDEX.md` for complete docs
- **Implementation Details**: See `IMPLEMENTATION_PLAN.md`
- **Project Structure**: See `PROJECT_STRUCTURE.md`

---

## Support

For issues and questions:
1. Check this guide's Troubleshooting section
2. Review `RISKS_AND_NOTES.md` for common problems
3. Check the documentation in the repository
4. Open an issue on GitHub (if applicable)

---

**Last Updated**: October 18, 2024
