#!/usr/bin/env python3
"""
Verification script to check if C2AE project is properly set up.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def check_imports():
    """Check if all required modules can be imported."""
    print("="*60)
    print("Checking module imports...")
    print("="*60)
    
    modules_to_check = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('numpy', 'NumPy'),
        ('sklearn', 'scikit-learn'),
        ('yaml', 'PyYAML'),
        ('matplotlib', 'Matplotlib'),
        ('tensorboard', 'TensorBoard'),
    ]
    
    all_good = True
    for module_name, display_name in modules_to_check:
        try:
            __import__(module_name)
            print(f"✅ {display_name:20s} - OK")
        except ImportError as e:
            print(f"❌ {display_name:20s} - MISSING")
            all_good = False
    
    return all_good


def check_project_structure():
    """Check if project structure is correct."""
    print("\n" + "="*60)
    print("Checking project structure...")
    print("="*60)
    
    required_dirs = [
        'src/models',
        'src/data',
        'src/training',
        'src/evaluation',
        'src/experiments',
        'src/utils',
        'configs',
        'tests',
        'scripts',
        'data',
        'results',
    ]
    
    all_good = True
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"✅ {dir_path:30s} - OK")
        else:
            print(f"❌ {dir_path:30s} - MISSING")
            all_good = False
    
    return all_good


def check_key_files():
    """Check if key files exist."""
    print("\n" + "="*60)
    print("Checking key files...")
    print("="*60)
    
    required_files = [
        'src/models/c2ae.py',
        'src/models/film_layers.py',
        'src/models/encoder.py',
        'src/models/decoder.py',
        'src/data/dataset_loader.py',
        'src/training/trainer.py',
        'src/evaluation/openset_evaluator.py',
        'src/experiments/run_experiment.py',
        'configs/default.yaml',
        'requirements.txt',
        'setup.py',
    ]
    
    all_good = True
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"✅ {file_path:40s} - OK")
        else:
            print(f"❌ {file_path:40s} - MISSING")
            all_good = False
    
    return all_good


def check_project_modules():
    """Check if project modules can be imported."""
    print("\n" + "="*60)
    print("Checking project modules...")
    print("="*60)
    
    modules_to_check = [
        ('src.models.c2ae', 'C2AE Model'),
        ('src.models.film_layers', 'FiLM Layers'),
        ('src.data.dataset_loader', 'Dataset Loader'),
        ('src.training.trainer', 'Trainer'),
        ('src.evaluation.openset_evaluator', 'Evaluator'),
        ('src.utils.config', 'Config Manager'),
    ]
    
    all_good = True
    for module_name, display_name in modules_to_check:
        try:
            __import__(module_name)
            print(f"✅ {display_name:30s} - OK")
        except Exception as e:
            print(f"❌ {display_name:30s} - ERROR: {str(e)[:40]}")
            all_good = False
    
    return all_good


def main():
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*10 + "C2AE Installation Verification" + " "*17 + "║")
    print("╚" + "="*58 + "╝")
    print()
    
    results = []
    
    # Check imports
    results.append(("Dependencies", check_imports()))
    
    # Check structure
    results.append(("Project Structure", check_project_structure()))
    
    # Check files
    results.append(("Key Files", check_key_files()))
    
    # Check project modules (only if dependencies are installed)
    if results[0][1]:
        results.append(("Project Modules", check_project_modules()))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_passed = True
    for check_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{check_name:30s}: {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n✅ All checks passed! Installation is complete.")
        print("\nNext steps:")
        print("  1. Download datasets: bash scripts/download_datasets.sh")
        print("  2. Run quick start: python examples/quick_start.py")
        print("  3. Train on MNIST: python -m src.experiments.run_experiment --config configs/mnist.yaml --mode train")
        return 0
    else:
        print("\n❌ Some checks failed. Please review the errors above.")
        if not results[0][1]:
            print("\nTo install dependencies, run:")
            print("  bash scripts/setup_environment.sh")
            print("  source venv/bin/activate")
        return 1


if __name__ == '__main__':
    sys.exit(main())
