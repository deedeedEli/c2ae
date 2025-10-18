"""
Quick start example for C2AE model.

This script demonstrates how to:
1. Load configuration
2. Create a C2AE model
3. Load data
4. Train the model
5. Evaluate on open-set recognition
"""

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.c2ae import C2AE
from src.data.dataset_loader import get_openset_loaders
from src.training.trainer import C2AETrainer
from src.evaluation.openset_evaluator import OpenSetEvaluator
from src.utils.config import Config
from src.utils.reproducibility import set_seed
from src.utils.device_manager import DeviceManager
from src.experiments.logger import ExperimentLogger


def main():
    # 1. Setup
    print("Setting up experiment...")
    config = Config('configs/mnist.yaml')
    set_seed(42)
    
    device_manager = DeviceManager(device='cuda' if torch.cuda.is_available() else 'cpu')
    device = device_manager.device
    print(f"Using device: {device}")
    
    # 2. Create model
    print("\nCreating C2AE model...")
    model = C2AE(
        num_classes=7,  # 6 known + 1 background
        embedding_dim=128,
        encoder_arch='custom',
        latent_dim=256,
        image_channels=1,  # MNIST is grayscale
        image_size=28,
        film_positions=[0, 1, 2, 3]
    )
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 3. Load data
    print("\nLoading MNIST data...")
    known_classes = [0, 1, 2, 3, 4, 5]  # First 6 digits
    data_loaders = get_openset_loaders(
        dataset_name='mnist',
        known_classes=known_classes,
        batch_size=256,
        config=config.config,
        num_workers=2,
        data_root='./data'
    )
    
    print(f"Train samples: {len(data_loaders['train'].dataset)}")
    print(f"Test known samples: {len(data_loaders['test_known'].dataset)}")
    print(f"Test unknown samples: {len(data_loaders['test_unknown'].dataset)}")
    
    # 4. Train (for just 5 epochs in this example)
    print("\nTraining model...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    logger = ExperimentLogger(experiment_name='c2ae_quickstart', use_tensorboard=False)
    
    trainer = C2AETrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config.config,
        logger=logger
    )
    
    # Train for just 5 epochs as a demo
    print("Training for 5 epochs (for full training, use 100+ epochs)...")
    history = trainer.fit(
        train_loader=data_loaders['train'],
        val_loader=data_loaders['test_known'],
        num_epochs=5
    )
    
    # 5. Evaluate
    print("\nEvaluating model...")
    evaluator = OpenSetEvaluator(
        model=model,
        known_classes=known_classes,
        device=device
    )
    
    results = evaluator.evaluate(
        test_known_loader=data_loaders['test_known'],
        test_unknown_loader=data_loaders['test_unknown']
    )
    
    print("\n" + "="*60)
    print("QUICK START RESULTS (5 epochs)")
    print("="*60)
    print(f"AUROC: {results['auroc']:.4f}")
    print(f"AUPR: {results['aupr']:.4f}")
    print(f"Known Class Accuracy: {results['known_classification_accuracy']:.4f}")
    print("="*60)
    print("\nNote: These are preliminary results from only 5 epochs.")
    print("For full performance, train for 100+ epochs using:")
    print("  python -m src.experiments.run_experiment --config configs/mnist.yaml --mode train")
    
    logger.close()


if __name__ == '__main__':
    main()
