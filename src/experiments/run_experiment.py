import argparse
import torch
import torch.optim as optim
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.c2ae import C2AE
from src.data.dataset_loader import get_openset_loaders
from src.training.trainer import C2AETrainer
from src.evaluation.openset_evaluator import OpenSetEvaluator
from src.utils.config import Config
from src.utils.reproducibility import set_seed
from src.utils.device_manager import DeviceManager
from src.experiments.logger import ExperimentLogger


def parse_args():
    parser = argparse.ArgumentParser(description='C2AE Open-Set Recognition Experiment')
    
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate', 'full'],
                       help='Execution mode')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint for evaluation')
    
    # Override config parameters
    parser.add_argument('--dataset', type=str, default=None,
                       help='Dataset name')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=None,
                       help='Number of epochs')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load configuration
    config = Config(args.config if args.config else None)
    
    # Apply command-line overrides
    overrides = {}
    if args.dataset:
        overrides['data'] = overrides.get('data', {})
        overrides['data']['dataset'] = args.dataset
    if args.batch_size:
        overrides['training'] = overrides.get('training', {})
        overrides['training']['batch_size'] = args.batch_size
    if args.learning_rate:
        overrides['training'] = overrides.get('training', {})
        overrides['training']['learning_rate'] = args.learning_rate
    if args.num_epochs:
        overrides['training'] = overrides.get('training', {})
        overrides['training']['num_epochs'] = args.num_epochs
    if args.seed:
        overrides['experiment'] = overrides.get('experiment', {})
        overrides['experiment']['seed'] = args.seed
    
    if overrides:
        config.update(overrides)
    
    # Set random seed
    seed = config.get('experiment.seed', 42)
    set_seed(seed, deterministic=config.get('hardware.cuda_deterministic', False))
    
    # Setup device
    device_manager = DeviceManager(
        device=config.get('hardware.device', 'cuda'),
        num_gpus=config.get('hardware.num_gpus', 1)
    )
    device = device_manager.device
    print(f"Using device: {device}")
    
    # Get dataset parameters
    dataset_name = config.get('data.dataset', 'cifar10')
    if dataset_name.lower() == 'mnist':
        image_size = 28
        image_channels = 1
        default_known_classes = [0, 1, 2, 3, 4, 5]
    elif dataset_name.lower() in ['svhn', 'cifar10']:
        image_size = 32
        image_channels = 3
        default_known_classes = [0, 1, 2, 3, 4, 5]
    else:
        image_size = 32
        image_channels = 3
        default_known_classes = [0, 1, 2, 3, 4, 5]
    
    known_classes = config.get('data.known_class_indices') or default_known_classes
    num_classes = len(known_classes) + 1  # +1 for background class
    
    # Create model
    print("Creating model...")
    model = C2AE(
        num_classes=num_classes,
        embedding_dim=config.get('model.embedding_dim', 128),
        encoder_arch=config.get('model.encoder_arch', 'resnet18'),
        latent_dim=config.get('model.latent_dim', 512),
        image_channels=image_channels,
        image_size=image_size,
        film_positions=config.get('model.film_positions', [0, 1, 2, 3])
    )
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load data
    print("Loading data...")
    data_loaders = get_openset_loaders(
        dataset_name=dataset_name,
        known_classes=known_classes,
        batch_size=config.get('training.batch_size', 128),
        config=config.config,
        num_workers=config.get('data.num_workers', 4),
        data_root=config.get('data.data_root', './data')
    )
    
    print(f"Train samples: {len(data_loaders['train'].dataset)}")
    print(f"Test known samples: {len(data_loaders['test_known'].dataset)}")
    print(f"Test unknown samples: {len(data_loaders['test_unknown'].dataset)}")
    
    if args.mode in ['train', 'full']:
        # Setup optimizer
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.get('training.learning_rate', 0.001),
            weight_decay=config.get('training.weight_decay', 0.0001)
        )
        
        # Setup scheduler
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get('training.scheduler_params.step_size', 30),
            gamma=config.get('training.scheduler_params.gamma', 0.1)
        )
        
        # Setup logger
        logger = ExperimentLogger(
            experiment_name=config.get('experiment.name', 'c2ae_default'),
            use_tensorboard=config.get('experiment.use_tensorboard', True)
        )
        
        # Log hyperparameters
        logger.log_hyperparameters(config.config)
        
        # Create trainer
        trainer = C2AETrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            config=config.config,
            logger=logger
        )
        
        # Train
        print("\nStarting training...")
        history = trainer.fit(
            train_loader=data_loaders['train'],
            val_loader=data_loaders['test_known'],
            num_epochs=config.get('training.num_epochs', 100)
        )
        
        print("\nTraining completed!")
        logger.close()
        
        # Load best model for evaluation
        if args.mode == 'full':
            best_checkpoint = trainer.checkpoint_manager.get_best_checkpoint()
            if best_checkpoint:
                checkpoint = torch.load(best_checkpoint, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded best model from epoch {checkpoint['epoch']}")
    
    if args.mode in ['evaluate', 'full']:
        # Load checkpoint if provided
        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from {args.checkpoint}")
        
        # Evaluate
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
        
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"AUROC: {results['auroc']:.4f}")
        print(f"AUPR: {results['aupr']:.4f}")
        print(f"Known Class Accuracy: {results['known_classification_accuracy']:.4f}")
        print("="*50)
        
        # Save results
        import json
        results_dir = Path(config.get('experiment.save_dir', './results')) / 'metrics'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            else:
                return obj
        
        import numpy as np
        results_serializable = convert_to_serializable(results)
        
        with open(results_dir / 'evaluation_results.json', 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"\nResults saved to {results_dir / 'evaluation_results.json'}")


if __name__ == '__main__':
    main()
