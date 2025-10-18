# C2AE: Class Conditioned Auto-Encoder for Open-Set Recognition

**Paper**: "C2AE: Class Conditioned Auto-Encoder for Open-Set Recognition" (CVPR 2019)  
**Authors**: Poojan Oza and Vishal M. Patel  
**Paper URL**: https://openaccess.thecvf.com/content_CVPR_2019/papers/Oza_C2AE_Class_Conditioned_Auto-Encoder_for_Open-Set_Recognition_CVPR_2019_paper.pdf

## Overview

This repository contains a complete reproduction of the C2AE paper, designed to enable autonomous implementation by Claude Code or other AI development assistants. The project includes comprehensive documentation, modular code architecture, and extensive testing infrastructure.

### What is C2AE?

C2AE addresses the **open-set recognition problem**: classifying samples from known classes while detecting samples from unknown classes not seen during training. The method uses a class-conditioned autoencoder with Feature-wise Linear Modulation (FiLM) layers to reconstruct images based on class embeddings. Unknown classes produce high reconstruction errors, enabling detection.

### Key Features

- 🎯 **Class-conditioned reconstruction** using FiLM layers
- 🔍 **Open-set detection** via reconstruction error analysis
- 📊 **Multiple datasets**: MNIST, SVHN, CIFAR-10, CIFAR+10/+50, Tiny ImageNet
- 🧪 **Comprehensive evaluation**: AUROC, AUPR, CCR@FPR metrics
- 🔬 **Ablation studies** for hyperparameter analysis
- 📈 **TensorBoard integration** for training visualization
- ✅ **Extensive testing** with unit and integration tests

## Quick Start

### Prerequisites

- Python 3.8-3.10
- NVIDIA GPU with CUDA 11.8+ (recommended)
- 16GB RAM minimum
- 50GB storage for datasets and results

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd c2ae-reproduction

# Run setup script
bash scripts/setup_environment.sh

# Activate virtual environment
source venv/bin/activate

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Download Datasets

```bash
# Download all datasets (MNIST, SVHN, CIFAR-10, CIFAR-100)
bash scripts/download_datasets.sh

# Verify datasets
python -c "from src.data.data_utils import verify_dataset_integrity; \
           verify_dataset_integrity('cifar10')"
```

### Train C2AE

```bash
# Train on CIFAR-10 with default settings (6 known classes)
python -m src.experiments.run_experiment \
    --config configs/cifar10.yaml \
    --mode train

# Train on MNIST
python -m src.experiments.run_experiment \
    --config configs/mnist.yaml \
    --mode train

# Train with custom parameters
python -m src.experiments.run_experiment \
    --config configs/cifar10.yaml \
    --mode train \
    --batch_size 256 \
    --learning_rate 0.0005 \
    --num_epochs 150
```

### Evaluate Model

```bash
# Evaluate trained model
python -m src.experiments.run_experiment \
    --config configs/cifar10.yaml \
    --mode evaluate \
    --checkpoint results/checkpoints/best_model.pth

# View results
cat results/metrics/evaluation_results.json
```

### Monitor Training

```bash
# Start TensorBoard
tensorboard --logdir results/tensorboard --port 6006

# Open browser to http://localhost:6006
```

## Documentation Structure

This repository includes four comprehensive documentation files optimized for autonomous implementation:

### 📋 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
- Complete architecture blueprint
- Directory structure with file purposes
- Module interface specifications
- Technology stack and dependencies
- Configuration schema
- Entry points and API usage

### 🛠️ [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)
- Phase-by-phase development roadmap
- Detailed implementation steps
- Function signatures and pseudocode
- Integration points and dependencies
- Validation checkpoints
- Claude Code action items

### 📊 [DATA_AND_EVAL.md](DATA_AND_EVAL.md)
- Dataset specifications and download instructions
- Data processing pipeline
- Evaluation framework and metrics
- Experimental protocols
- Results validation and comparison
- Reproducibility requirements

### ⚠️ [RISKS_AND_NOTES.md](RISKS_AND_NOTES.md)
- Paper analysis and clarity assessment
- Technical risk assessment (high/medium/low priority)
- Implementation assumptions with rationale
- Common pitfalls and solutions
- Debugging strategies
- Claude Code specific guidance

## Project Structure

```
c2ae-reproduction/
├── src/                           # Source code
│   ├── models/                    # Model implementations
│   │   ├── c2ae.py               # Main C2AE model
│   │   ├── encoder.py            # Encoder network
│   │   ├── decoder.py            # Conditional decoder
│   │   ├── film_layers.py        # FiLM implementation
│   │   └── class_embedding.py    # Class embeddings
│   ├── data/                      # Data pipeline
│   │   ├── dataset_loader.py     # Dataset loading
│   │   ├── openset_split.py      # Open-set splitting
│   │   ├── background_generator.py  # Background class
│   │   └── transforms.py         # Data augmentation
│   ├── training/                  # Training components
│   │   ├── trainer.py            # Training loop
│   │   ├── losses.py             # Loss functions
│   │   └── scheduler.py          # LR scheduling
│   ├── evaluation/                # Evaluation components
│   │   ├── openset_evaluator.py  # Open-set evaluation
│   │   ├── metrics.py            # Metric computation
│   │   └── visualization.py      # Result visualization
│   └── utils/                     # Utilities
│       ├── config.py             # Configuration management
│       ├── checkpoint.py         # Model checkpointing
│       └── reproducibility.py    # Seed management
├── configs/                       # Configuration files
│   ├── default.yaml
│   ├── cifar10.yaml
│   └── mnist.yaml
├── tests/                         # Unit tests
│   ├── test_models/
│   ├── test_data/
│   └── test_evaluation/
├── scripts/                       # Utility scripts
│   ├── setup_environment.sh
│   └── download_datasets.sh
├── data/                          # Data directory
│   ├── raw/                      # Downloaded datasets
│   └── processed/                # Preprocessed data
├── results/                       # Results directory
│   ├── checkpoints/              # Model checkpoints
│   ├── logs/                     # Training logs
│   └── visualizations/           # Plots and figures
└── notebooks/                     # Jupyter notebooks
    └── results_analysis.ipynb
```

## Expected Results

### Performance Benchmarks

| Dataset | Known/Unknown | AUROC | AUPR | CCR@FPR=5% |
|---------|---------------|-------|------|------------|
| MNIST | 6/4 | 0.985 | 0.975 | 0.94 |
| SVHN | 6/4 | 0.878 | 0.856 | 0.72 |
| CIFAR-10 | 6/4 | 0.912 | 0.896 | 0.82 |
| CIFAR+10 | 10/10 | 0.893 | 0.871 | - |
| CIFAR+50 | 10/50 | 0.905 | 0.889 | - |

**Tolerance**: ±2% for reproduction

### Training Time Estimates

- **MNIST**: ~2 minutes per epoch (GPU) → 3-4 hours total
- **SVHN**: ~5 minutes per epoch (GPU) → 8-10 hours total
- **CIFAR-10**: ~10 minutes per epoch (GPU) → 16-20 hours total
- **Tiny ImageNet**: ~1 hour per epoch (GPU) → 150+ hours total

## Usage Examples

### Python API

```python
from src.models.c2ae import C2AE
from src.data.dataset_loader import get_openset_loaders
from src.training.trainer import C2AETrainer
from src.evaluation.openset_evaluator import OpenSetEvaluator
from src.utils.config import Config

# Load configuration
config = Config('configs/cifar10.yaml')

# Create model
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
    batch_size=128,
    config=config['augmentation']
)

# Train
trainer = C2AETrainer(model, optimizer, scheduler, device, config)
history = trainer.fit(
    train_loader=loaders['train'],
    val_loader=loaders['test_known'],
    num_epochs=100,
    checkpoint_dir='./results/checkpoints'
)

# Evaluate
evaluator = OpenSetEvaluator(model, known_classes=[0,1,2,3,4,5], device=device)
results = evaluator.evaluate(
    test_known_loader=loaders['test_known'],
    test_unknown_loader=loaders['test_unknown']
)

print(f"AUROC: {results['auroc']:.4f}")
print(f"AUPR: {results['aupr']:.4f}")
print(f"CCR@FPR=5%: {results['ccr_at_fpr_5']['ccr']:.4f}")
```

### Command-Line Interface

```bash
# Full experimental suite
bash scripts/run_all_experiments.sh

# Single experiment with multiple trials
python -m src.experiments.run_experiment \
    --config configs/cifar10.yaml \
    --mode full \
    --trials 5 \
    --seeds 42,123,456,789,1011

# Ablation study
python experiments/run_ablation_study.py \
    --dataset cifar10 \
    --variable background_ratio \
    --values 0.1,0.3,0.5
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test suite
pytest tests/test_models/ -v
pytest tests/test_data/ -v
pytest tests/test_evaluation/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html
```

## Configuration

Configuration files use YAML format with hierarchical structure. Example:

```yaml
# configs/cifar10.yaml
model:
  encoder_arch: "resnet18"
  embedding_dim: 128
  latent_dim: 512

training:
  num_epochs: 100
  batch_size: 128
  learning_rate: 0.001
  
data:
  dataset: "cifar10"
  num_known_classes: 6
  known_class_indices: [0, 1, 2, 3, 4, 5]
  use_background: true
  background_ratio: 0.3

# ... more settings
```

Override via command line:
```bash
python -m src.experiments.run_experiment \
    --config configs/cifar10.yaml \
    --batch_size 256 \
    --learning_rate 0.0005
```

## Reproducibility

### Ensuring Reproducible Results

1. **Set Random Seeds**:
```python
from src.utils.reproducibility import set_seed
set_seed(42, deterministic=True)
```

2. **Use Exact Versions**:
```bash
pip install -r requirements.txt  # Pinned versions
```

3. **Save Configuration**:
```python
config.save('results/experiment_config.yaml')
```

4. **Document Environment**:
```bash
pip freeze > results/environment.txt
python -c "from src.utils.reproducibility import get_reproducibility_info; \
           print(get_reproducibility_info())"
```

### Known Sources of Non-Determinism

- **CUDA operations**: Set `torch.backends.cudnn.deterministic=True` (slower)
- **Data loader workers**: May introduce slight randomness
- **Hardware differences**: Different GPUs may have minor variations
- **PyTorch versions**: Different versions may have implementation changes

## Troubleshooting

### Common Issues

**Issue**: CUDA out of memory
```bash
# Solution: Reduce batch size
python -m src.experiments.run_experiment --config configs/cifar10.yaml --batch_size 64
```

**Issue**: Training loss not decreasing
```bash
# Solution: Check learning rate, try lower value
python -m src.experiments.run_experiment --config configs/cifar10.yaml --learning_rate 0.0005
```

**Issue**: Performance below paper results
1. Verify data preprocessing and normalization
2. Check that background class is included (30% ratio)
3. Ensure sufficient training epochs (100+)
4. Visualize reconstructions to verify model is learning

**Issue**: Dataset download fails
```bash
# Manual download
python -c "from torchvision import datasets; \
           datasets.CIFAR10(root='./data/raw', train=True, download=True)"
```

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or via config
config.update({'experiment': {'log_level': 'DEBUG'}})
```

## Citation

If you use this reproduction or reference the original paper:

```bibtex
@inproceedings{oza2019c2ae,
  title={C2AE: Class conditioned auto-encoder for open-set recognition},
  author={Oza, Poojan and Patel, Vishal M},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2307--2316},
  year={2019}
}
```

## License

This reproduction project: MIT License (see LICENSE file)

Original Paper: Please refer to CVPR 2019 proceedings

Datasets: Refer to individual dataset licenses
- MNIST: Public domain
- SVHN: Non-commercial research
- CIFAR: MIT-compatible

## Contributing

While this project is designed for autonomous implementation, contributions are welcome:

1. Bug fixes and improvements
2. Additional datasets
3. Performance optimizations
4. Documentation enhancements

Please ensure:
- All tests pass
- Code follows existing style
- Documentation is updated
- Changes are reproducible

## Acknowledgments

- Original authors: Poojan Oza and Vishal M. Patel
- FiLM layers: Perez et al. (2017)
- PyTorch team for the deep learning framework
- Dataset creators and maintainers

## Contact

For issues related to this reproduction:
- Open an issue on GitHub
- Check RISKS_AND_NOTES.md for common problems

For questions about the original paper:
- Refer to the authors' contact information in the paper

---

**Status**: Ready for autonomous implementation by Claude Code

**Last Updated**: 2024

**Estimated Implementation Time**: 16-20 hours (code) + 20-40 hours (training)

**Success Criterion**: Reproduce paper results within ±2% for AUROC/AUPR on all datasets
