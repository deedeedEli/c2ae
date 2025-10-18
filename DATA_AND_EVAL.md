# DATA_AND_EVAL.md - Data and Evaluation Specifications

## Research Context

**Research Question**: Can a class-conditioned autoencoder effectively distinguish between known and unknown classes in open-set recognition scenarios?

**Core Hypothesis**: By training an autoencoder to reconstruct images conditioned on class labels, the model will produce high reconstruction errors for unknown classes (since it hasn't learned appropriate class embeddings for them), enabling effective open-set recognition.

**Results to Reproduce**:
1. **Primary Metrics**: AUROC and AUPR for unknown detection across multiple datasets
2. **Closed-Set Accuracy**: Classification accuracy on known classes
3. **Open-Set Performance**: CCR (Correct Classification Rate) at FPR=5%
4. **Comparative Analysis**: Performance comparison with OpenMax, CROSR, and other baselines

**Expected Performance Ranges** (from paper):
- CIFAR-10 (6 known/4 unknown): AUROC ~0.90-0.93, AUPR ~0.88-0.91
- MNIST (6 known/4 unknown): AUROC ~0.98-0.99, AUPR ~0.97-0.98
- SVHN (6 known/4 unknown): AUROC ~0.85-0.90, AUPR ~0.82-0.88
- Tiny ImageNet: AUROC ~0.75-0.82

---

## Dataset Specifications

### 1. MNIST (Modified National Institute of Standards and Technology)

**Source**: http://yann.lecun.com/exdb/mnist/
**Description**: Handwritten digits (0-9)

**Download Command**:
```python
from torchvision import datasets
datasets.MNIST(root='./data/raw', train=True, download=True)
datasets.MNIST(root='./data/raw', train=False, download=True)
```

**Statistics**:
- **Total Classes**: 10
- **Training Samples**: 60,000
- **Test Samples**: 10,000
- **Image Size**: 28×28 pixels
- **Channels**: 1 (grayscale)
- **File Size**: ~50 MB

**License**: Public domain

**Open-Set Splits** (as per paper):
- **Protocol 1**: 6 known classes, 4 unknown
  - Known: [0, 1, 2, 3, 4, 5]
  - Unknown: [6, 7, 8, 9]
- **Protocol 2**: 4 known classes, 6 unknown
  - Known: [0, 1, 2, 3]
  - Unknown: [4, 5, 6, 7, 8, 9]

**Expected Performance**:
- 6 known/4 unknown: AUROC=0.985, AUPR=0.975, CCR@FPR5%=0.94

### 2. SVHN (Street View House Numbers)

**Source**: http://ufldl.stanford.edu/housenumbers/
**Description**: Real-world house numbers from Google Street View

**Download Command**:
```python
from torchvision import datasets
datasets.SVHN(root='./data/raw', split='train', download=True)
datasets.SVHN(root='./data/raw', split='test', download=True)
```

**Statistics**:
- **Total Classes**: 10 (digits 0-9)
- **Training Samples**: 73,257
- **Test Samples**: 26,032
- **Image Size**: 32×32 pixels
- **Channels**: 3 (RGB)
- **File Size**: ~1.5 GB

**License**: Non-commercial research use

**Open-Set Splits**:
- **Protocol 1**: 6 known classes, 4 unknown
  - Known: [0, 1, 2, 3, 4, 5]
  - Unknown: [6, 7, 8, 9]

**Expected Performance**:
- 6 known/4 unknown: AUROC=0.878, AUPR=0.856, CCR@FPR5%=0.72

### 3. CIFAR-10

**Source**: https://www.cs.toronto.edu/~kriz/cifar.html
**Description**: 10 classes of natural images

**Download Command**:
```python
from torchvision import datasets
datasets.CIFAR10(root='./data/raw', train=True, download=True)
datasets.CIFAR10(root='./data/raw', train=False, download=True)
```

**Statistics**:
- **Total Classes**: 10
  - airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- **Training Samples**: 50,000 (5,000 per class)
- **Test Samples**: 10,000 (1,000 per class)
- **Image Size**: 32×32 pixels
- **Channels**: 3 (RGB)
- **File Size**: ~170 MB

**License**: MIT-compatible

**Open-Set Splits** (multiple protocols from paper):
- **Protocol 1**: 6 known classes, 4 unknown
  - Known: [0, 1, 2, 3, 4, 5] (airplane, automobile, bird, cat, deer, dog)
  - Unknown: [6, 7, 8, 9] (frog, horse, ship, truck)
- **Protocol 2**: 4 known classes, 6 unknown
  - Known: [0, 1, 2, 3]
  - Unknown: [4, 5, 6, 7, 8, 9]
- **Protocol 3**: Random splits with different known/unknown ratios

**Expected Performance**:
- 6 known/4 unknown: AUROC=0.912, AUPR=0.896, CCR@FPR5%=0.82
- 4 known/6 unknown: AUROC=0.865, AUPR=0.848

### 4. CIFAR+10 and CIFAR+50

**Description**: Combined CIFAR-10 and CIFAR-100 for cross-dataset evaluation
- **CIFAR+10**: CIFAR-10 as known, 10 CIFAR-100 classes as unknown
- **CIFAR+50**: CIFAR-10 as known, 50 CIFAR-100 classes as unknown

**Download Commands**:
```python
from torchvision import datasets
# CIFAR-10 (known classes)
datasets.CIFAR10(root='./data/raw', train=True, download=True)
datasets.CIFAR10(root='./data/raw', train=False, download=True)

# CIFAR-100 (unknown classes)
datasets.CIFAR100(root='./data/raw', train=True, download=True)
datasets.CIFAR100(root='./data/raw', train=False, download=True)
```

**Statistics**:
- **CIFAR-10**: 10 classes, 60,000 images
- **CIFAR-100**: 100 classes, 60,000 images
- **Image Size**: 32×32 pixels
- **Channels**: 3 (RGB)

**Open-Set Configuration**:
- Train on all CIFAR-10 classes (10 known)
- Test with CIFAR-10 (known) + subset of CIFAR-100 (unknown)

**Expected Performance**:
- CIFAR+10: AUROC=0.893, AUPR=0.871
- CIFAR+50: AUROC=0.905, AUPR=0.889

### 5. Tiny ImageNet

**Source**: http://cs231n.stanford.edu/tiny-imagenet-200.zip
**Description**: Subset of ImageNet with 200 classes

**Download Command**:
```bash
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip -d ./data/raw/
```

**Statistics**:
- **Total Classes**: 200
- **Training Samples**: 100,000 (500 per class)
- **Validation Samples**: 10,000 (50 per class)
- **Image Size**: 64×64 pixels
- **Channels**: 3 (RGB)
- **File Size**: ~240 MB

**License**: Academic use

**Open-Set Splits**:
- **Protocol 1**: 20 known classes, 180 unknown
- **Protocol 2**: 50 known classes, 150 unknown

**Expected Performance**:
- 20 known/180 unknown: AUROC=0.782, AUPR=0.445

---

## Data Processing Pipeline

### Step 1: Dataset Download and Verification

**CLAUDE CODE TASK**: Implement in `scripts/download_datasets.sh`

```bash
#!/bin/bash
# Download all required datasets

DATA_DIR="./data/raw"
mkdir -p $DATA_DIR

echo "Downloading datasets for C2AE reproduction..."

# Download via Python
python3 << EOF
from torchvision import datasets
import os

data_dir = '$DATA_DIR'

print("Downloading MNIST...")
datasets.MNIST(root=data_dir, train=True, download=True)
datasets.MNIST(root=data_dir, train=False, download=True)

print("Downloading SVHN...")
datasets.SVHN(root=data_dir, split='train', download=True)
datasets.SVHN(root=data_dir, split='test', download=True)

print("Downloading CIFAR-10...")
datasets.CIFAR10(root=data_dir, train=True, download=True)
datasets.CIFAR10(root=data_dir, train=False, download=True)

print("Downloading CIFAR-100...")
datasets.CIFAR100(root=data_dir, train=True, download=True)
datasets.CIFAR100(root=data_dir, train=False, download=True)

print("All downloads complete!")
EOF

# Download Tiny ImageNet
if [ ! -d "$DATA_DIR/tiny-imagenet-200" ]; then
    echo "Downloading Tiny ImageNet..."
    cd $DATA_DIR
    wget -q http://cs231n.stanford.edu/tiny-imagenet-200.zip
    unzip -q tiny-imagenet-200.zip
    rm tiny-imagenet-200.zip
    cd -
fi

echo "Dataset download complete!"
```

**Verification Script** (`src/data/data_utils.py`):

```python
def verify_dataset_integrity(dataset_name: str, data_root: str = './data') -> bool:
    """
    Verify dataset is correctly downloaded and accessible.
    
    Args:
        dataset_name: Name of dataset to verify
        data_root: Root data directory
    
    Returns:
        True if dataset is valid
    """
    try:
        if dataset_name == 'mnist':
            train_data = datasets.MNIST(root=f'{data_root}/raw', train=True)
            test_data = datasets.MNIST(root=f'{data_root}/raw', train=False)
            assert len(train_data) == 60000
            assert len(test_data) == 10000
            
        elif dataset_name == 'svhn':
            train_data = datasets.SVHN(root=f'{data_root}/raw', split='train')
            test_data = datasets.SVHN(root=f'{data_root}/raw', split='test')
            assert len(train_data) == 73257
            assert len(test_data) == 26032
            
        elif dataset_name == 'cifar10':
            train_data = datasets.CIFAR10(root=f'{data_root}/raw', train=True)
            test_data = datasets.CIFAR10(root=f'{data_root}/raw', train=False)
            assert len(train_data) == 50000
            assert len(test_data) == 10000
        
        print(f"✓ {dataset_name} verified successfully")
        return True
        
    except Exception as e:
        print(f"✗ {dataset_name} verification failed: {e}")
        return False
```

### Step 2: Data Preprocessing

**Normalization Strategy**:
- **MNIST**: Mean=[0.5], Std=[0.5] (single channel)
- **SVHN/CIFAR**: Mean=[0.5, 0.5, 0.5], Std=[0.5, 0.5, 0.5]
- **Tiny ImageNet**: Mean=[0.485, 0.456, 0.406], Std=[0.229, 0.224, 0.225] (ImageNet stats)

**Implementation** (`src/data/transforms.py` - already implemented in Phase 2):

```python
NORMALIZATION_STATS = {
    'mnist': {
        'mean': [0.5],
        'std': [0.5]
    },
    'svhn': {
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5]
    },
    'cifar10': {
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5]
    },
    'cifar100': {
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5]
    },
    'tiny_imagenet': {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }
}
```

### Step 3: Open-Set Split Generation

**Implementation** (`src/data/openset_split.py` - already covered in Phase 2):

**Usage Example**:
```python
from src.data.openset_split import OpenSetSplitter

# Create splitter for CIFAR-10 with 6 known classes
splitter = OpenSetSplitter(
    num_total_classes=10,
    num_known_classes=6,
    known_class_indices=[0, 1, 2, 3, 4, 5],
    seed=42
)

# Generate split
split_dict = splitter.split_dataset_indices(dataset_labels)

# Save for reproducibility
splitter.save_split(split_dict, './data/openset_splits/cifar10_6known.json')
```

### Step 4: Background Class Generation

**Strategy**: Use samples from unknown classes with strong augmentation to create out-of-distribution examples during training.

**Augmentation Techniques for Background**:
1. Random crops with larger padding (±8 pixels)
2. Random rotations (±30 degrees)
3. Color jittering (brightness, contrast, saturation, hue)
4. Random grayscale conversion
5. Gaussian blur
6. Random vertical flips (in addition to horizontal)

**Ratio**: 30% background samples relative to known class samples (paper specification)

**Implementation**: Already covered in `src/data/background_generator.py` (Phase 2)

### Step 5: Data Loading Strategy

**Batch Composition**:
- Training batches contain mix of known classes + background class
- Each training batch: ~70% known classes, ~30% background
- Test batches separated into known and unknown

**Data Loader Configuration**:
```python
# Training
- batch_size: 128
- shuffle: True
- num_workers: 4
- pin_memory: True
- drop_last: True  # For batch norm stability

# Testing
- batch_size: 128
- shuffle: False
- num_workers: 4
- pin_memory: True
- drop_last: False
```

---

## Evaluation Framework

### Primary Metrics

#### 1. AUROC (Area Under Receiver Operating Characteristic)

**Purpose**: Measure ability to distinguish known from unknown classes across all possible thresholds.

**Implementation** (`src/evaluation/metrics.py`):

```python
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np

def compute_auroc(known_scores: np.ndarray, 
                  unknown_scores: np.ndarray) -> float:
    """
    Compute AUROC for open-set detection.
    
    Args:
        known_scores: Reconstruction errors for known classes (lower is better)
        unknown_scores: Reconstruction errors for unknown classes (higher is better)
    
    Returns:
        AUROC value in [0, 1], where 1.0 is perfect
    """
    # Combine scores
    scores = np.concatenate([known_scores, unknown_scores])
    
    # Create labels (0=known, 1=unknown)
    labels = np.concatenate([
        np.zeros(len(known_scores)),
        np.ones(len(unknown_scores))
    ])
    
    # Since known should have lower scores, we want higher scores to indicate unknown
    # So we use scores directly (higher score = more likely unknown)
    auroc = roc_auc_score(labels, scores)
    
    return auroc


def compute_roc_curve(known_scores: np.ndarray,
                     unknown_scores: np.ndarray):
    """
    Compute full ROC curve.
    
    Returns:
        Tuple of (fpr, tpr, thresholds)
    """
    scores = np.concatenate([known_scores, unknown_scores])
    labels = np.concatenate([
        np.zeros(len(known_scores)),
        np.ones(len(unknown_scores))
    ])
    
    fpr, tpr, thresholds = roc_curve(labels, scores)
    return fpr, tpr, thresholds
```

**Interpretation**:
- AUROC = 1.0: Perfect separation
- AUROC = 0.5: Random guessing
- AUROC > 0.9: Excellent performance
- AUROC > 0.8: Good performance
- AUROC < 0.7: Poor performance

#### 2. AUPR (Area Under Precision-Recall Curve)

**Purpose**: Measure precision-recall trade-off, particularly important when unknown classes are rare.

**Implementation**:

```python
from sklearn.metrics import average_precision_score, precision_recall_curve

def compute_aupr(known_scores: np.ndarray,
                 unknown_scores: np.ndarray) -> float:
    """
    Compute AUPR for open-set detection.
    
    Args:
        known_scores: Reconstruction errors for known classes
        unknown_scores: Reconstruction errors for unknown classes
    
    Returns:
        AUPR value in [0, 1]
    """
    scores = np.concatenate([known_scores, unknown_scores])
    labels = np.concatenate([
        np.zeros(len(known_scores)),
        np.ones(len(unknown_scores))
    ])
    
    aupr = average_precision_score(labels, scores)
    return aupr


def compute_pr_curve(known_scores: np.ndarray,
                    unknown_scores: np.ndarray):
    """
    Compute precision-recall curve.
    
    Returns:
        Tuple of (precision, recall, thresholds)
    """
    scores = np.concatenate([known_scores, unknown_scores])
    labels = np.concatenate([
        np.zeros(len(known_scores)),
        np.ones(len(unknown_scores))
    ])
    
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    return precision, recall, thresholds
```

#### 3. CCR @ FPR (Correct Classification Rate at Fixed False Positive Rate)

**Purpose**: Measure classification accuracy on known classes at a specific false positive rate (typically 5%).

**Implementation**:

```python
def compute_ccr_at_fpr(known_scores: np.ndarray,
                       unknown_scores: np.ndarray,
                       known_predictions: np.ndarray,
                       known_labels: np.ndarray,
                       fpr_threshold: float = 0.05) -> tuple:
    """
    Compute CCR at specific FPR threshold.
    
    Args:
        known_scores: Reconstruction errors for known samples
        unknown_scores: Reconstruction errors for unknown samples
        known_predictions: Predicted classes for known samples
        known_labels: True labels for known samples
        fpr_threshold: Desired FPR (default: 0.05 for 5%)
    
    Returns:
        Tuple of (ccr, threshold, actual_fpr)
    """
    # Get ROC curve
    fpr, tpr, thresholds = compute_roc_curve(known_scores, unknown_scores)
    
    # Find threshold closest to desired FPR
    idx = np.argmin(np.abs(fpr - fpr_threshold))
    threshold = thresholds[idx]
    actual_fpr = fpr[idx]
    
    # Classify known samples based on threshold
    known_mask = known_scores < threshold
    
    # Compute accuracy only on samples classified as known
    if known_mask.sum() > 0:
        correct = (known_predictions[known_mask] == known_labels[known_mask]).sum()
        ccr = correct / known_mask.sum()
    else:
        ccr = 0.0
    
    return ccr, threshold, actual_fpr


def compute_ccr_at_multiple_fprs(known_scores: np.ndarray,
                                 unknown_scores: np.ndarray,
                                 known_predictions: np.ndarray,
                                 known_labels: np.ndarray,
                                 fpr_thresholds: list = [0.01, 0.05, 0.1]) -> dict:
    """
    Compute CCR at multiple FPR thresholds.
    
    Returns:
        Dictionary mapping FPR to (CCR, threshold)
    """
    results = {}
    for fpr_thresh in fpr_thresholds:
        ccr, threshold, actual_fpr = compute_ccr_at_fpr(
            known_scores, unknown_scores,
            known_predictions, known_labels,
            fpr_thresh
        )
        results[f'ccr_at_fpr_{int(fpr_thresh*100)}'] = {
            'ccr': ccr,
            'threshold': threshold,
            'actual_fpr': actual_fpr
        }
    
    return results
```

#### 4. F1 Score at Optimal Threshold

**Purpose**: Measure balanced performance at automatically selected threshold.

**Implementation**:

```python
from sklearn.metrics import f1_score

def compute_optimal_f1(known_scores: np.ndarray,
                      unknown_scores: np.ndarray) -> tuple:
    """
    Find optimal threshold by maximizing F1 score.
    
    Returns:
        Tuple of (best_f1, optimal_threshold)
    """
    scores = np.concatenate([known_scores, unknown_scores])
    labels = np.concatenate([
        np.zeros(len(known_scores)),
        np.ones(len(unknown_scores))
    ])
    
    # Try different thresholds
    thresholds = np.percentile(scores, np.linspace(0, 100, 1000))
    best_f1 = 0.0
    optimal_threshold = 0.0
    
    for threshold in thresholds:
        predictions = (scores >= threshold).astype(int)
        f1 = f1_score(labels, predictions)
        
        if f1 > best_f1:
            best_f1 = f1
            optimal_threshold = threshold
    
    return best_f1, optimal_threshold
```

### Secondary Metrics

#### 5. Closed-Set Accuracy

**Purpose**: Measure classification accuracy on known classes (ignoring unknown detection).

```python
def compute_closed_set_accuracy(predictions: np.ndarray,
                               labels: np.ndarray) -> float:
    """
    Compute standard classification accuracy.
    
    Args:
        predictions: Predicted class labels
        labels: True class labels
    
    Returns:
        Accuracy in [0, 1]
    """
    correct = (predictions == labels).sum()
    total = len(labels)
    return correct / total if total > 0 else 0.0
```

#### 6. FPR at 95% TPR

**Purpose**: Measure false positive rate when detecting 95% of unknowns.

```python
def compute_fpr_at_tpr(known_scores: np.ndarray,
                      unknown_scores: np.ndarray,
                      tpr_threshold: float = 0.95) -> tuple:
    """
    Compute FPR at specific TPR.
    
    Returns:
        Tuple of (fpr, threshold)
    """
    fpr, tpr, thresholds = compute_roc_curve(known_scores, unknown_scores)
    
    # Find threshold for desired TPR
    idx = np.argmin(np.abs(tpr - tpr_threshold))
    return fpr[idx], thresholds[idx]
```

### Comprehensive Evaluator

**CLAUDE CODE TASK**: Implement in `src/evaluation/openset_evaluator.py`

```python
class OpenSetEvaluator:
    """Complete open-set evaluation suite."""
    
    def __init__(self, model, known_classes, device):
        self.model = model
        self.known_classes = known_classes
        self.device = device
        self.model.eval()
    
    def evaluate(self, test_known_loader, test_unknown_loader):
        """
        Comprehensive open-set evaluation.
        
        Returns:
            Dictionary with all metrics
        """
        # Collect scores and predictions
        known_scores, known_preds, known_labels = self._compute_scores(test_known_loader, is_known=True)
        unknown_scores, _, _ = self._compute_scores(test_unknown_loader, is_known=False)
        
        # Compute metrics
        results = {}
        
        # Primary metrics
        results['auroc'] = compute_auroc(known_scores, unknown_scores)
        results['aupr'] = compute_aupr(known_scores, unknown_scores)
        
        # F1 score
        best_f1, optimal_threshold = compute_optimal_f1(known_scores, unknown_scores)
        results['f1_score'] = best_f1
        results['optimal_threshold'] = optimal_threshold
        
        # CCR at multiple FPRs
        ccr_results = compute_ccr_at_multiple_fprs(
            known_scores, unknown_scores,
            known_preds, known_labels,
            fpr_thresholds=[0.01, 0.05, 0.1]
        )
        results.update(ccr_results)
        
        # FPR at 95% TPR
        fpr_at_95, _ = compute_fpr_at_tpr(known_scores, unknown_scores, tpr_threshold=0.95)
        results['fpr_at_95_tpr'] = fpr_at_95
        
        # Closed-set accuracy
        results['closed_set_accuracy'] = compute_closed_set_accuracy(known_preds, known_labels)
        
        # Per-class accuracy
        results['per_class_accuracy'] = self._compute_per_class_accuracy(known_preds, known_labels)
        
        return results
    
    def _compute_scores(self, data_loader, is_known=True):
        """Compute reconstruction scores for dataset."""
        all_scores = []
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Computing scores"):
                images, labels, _ = batch
                images = images.to(self.device)
                
                # Get predictions and scores
                result = self.model.predict_openset(
                    images,
                    known_classes=self.known_classes,
                    threshold=None,
                    return_scores=True
                )
                
                all_scores.append(result['min_scores'].cpu().numpy())
                all_preds.append(result['predictions'].cpu().numpy())
                all_labels.append(labels.numpy())
        
        scores = np.concatenate(all_scores)
        preds = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)
        
        return scores, preds, labels
```

---

## Experimental Protocols

### Protocol 1: Standard Open-Set Recognition

**Configuration**:
- **Dataset**: CIFAR-10
- **Known Classes**: 6 (classes 0-5)
- **Unknown Classes**: 4 (classes 6-9)
- **Training**: 80% of known class samples + 30% background
- **Testing**: Remaining 20% of known + all unknown

**Training Hyperparameters**:
```yaml
training:
  num_epochs: 100
  batch_size: 128
  learning_rate: 0.001
  weight_decay: 0.0001
  optimizer: adam
  scheduler: step
  scheduler_params:
    step_size: 30
    gamma: 0.1
```

**Expected Runtime**: ~10 minutes per epoch on GPU, ~1 hour total

### Protocol 2: Cross-Dataset Evaluation (CIFAR+10)

**Configuration**:
- **Known**: All CIFAR-10 classes (10 classes)
- **Unknown**: 10 randomly selected CIFAR-100 classes
- **Training**: All CIFAR-10 training data + background from CIFAR-100
- **Testing**: CIFAR-10 test + selected CIFAR-100 test samples

### Protocol 3: Large-Scale (Tiny ImageNet)

**Configuration**:
- **Known Classes**: 20 classes
- **Unknown Classes**: 180 classes
- **Image Size**: 64×64 (larger than CIFAR)
- **Training Epochs**: 150 (larger dataset requires more epochs)

**Modified Hyperparameters**:
```yaml
training:
  num_epochs: 150
  batch_size: 64  # Reduced due to larger images
  learning_rate: 0.0005  # Lower LR for larger model
```

### Protocol 4: Few-Shot Open-Set

**Configuration**:
- Limited training samples per class (100, 500, 1000)
- Evaluate robustness with reduced data
- Expected performance degradation: ~5-10% AUROC drop

---

## Results Validation

### Automated Comparison

**CLAUDE CODE TASK**: Implement in `src/evaluation/result_validator.py`

```python
class ResultValidator:
    """Validate results against paper benchmarks."""
    
    def __init__(self):
        # Paper results (Table 1 from C2AE paper)
        self.paper_results = {
            'cifar10_6known': {
                'auroc': 0.912,
                'aupr': 0.896,
                'ccr_at_fpr_5': 0.82,
                'tolerance': 0.02  # ±2% tolerance
            },
            'mnist_6known': {
                'auroc': 0.985,
                'aupr': 0.975,
                'ccr_at_fpr_5': 0.94,
                'tolerance': 0.02
            },
            'svhn_6known': {
                'auroc': 0.878,
                'aupr': 0.856,
                'ccr_at_fpr_5': 0.72,
                'tolerance': 0.03
            }
        }
    
    def validate_results(self, experiment_name: str, results: dict) -> dict:
        """
        Compare results with paper benchmarks.
        
        Returns:
            Dictionary with validation status and differences
        """
        if experiment_name not in self.paper_results:
            return {'status': 'no_benchmark', 'message': 'No benchmark available'}
        
        benchmark = self.paper_results[experiment_name]
        tolerance = benchmark['tolerance']
        
        validation = {
            'status': 'passed',
            'differences': {},
            'issues': []
        }
        
        for metric in ['auroc', 'aupr', 'ccr_at_fpr_5']:
            if metric in results:
                paper_value = benchmark[metric]
                repro_value = results[metric]
                diff = repro_value - paper_value
                
                validation['differences'][metric] = {
                    'paper': paper_value,
                    'reproduction': repro_value,
                    'difference': diff,
                    'within_tolerance': abs(diff) <= tolerance
                }
                
                if abs(diff) > tolerance:
                    validation['status'] = 'failed'
                    validation['issues'].append(
                        f"{metric}: {repro_value:.4f} vs paper {paper_value:.4f} "
                        f"(diff: {diff:+.4f}, tolerance: ±{tolerance})"
                    )
        
        return validation
```

### Visualization and Reporting

**CLAUDE CODE TASK**: Implement in `src/evaluation/visualization.py`

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_roc_curve(known_scores, unknown_scores, save_path=None):
    """Plot ROC curve for open-set detection."""
    fpr, tpr, _ = compute_roc_curve(known_scores, unknown_scores)
    auroc = compute_auroc(known_scores, unknown_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'C2AE (AUROC={auroc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Open-Set Detection', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_score_distribution(known_scores, unknown_scores, save_path=None):
    """Plot distribution of reconstruction scores."""
    plt.figure(figsize=(10, 6))
    
    plt.hist(known_scores, bins=50, alpha=0.6, label='Known', density=True)
    plt.hist(unknown_scores, bins=50, alpha=0.6, label='Unknown', density=True)
    
    plt.xlabel('Reconstruction Error', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Score Distribution', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_results_table(results_dict, save_path=None):
    """Generate formatted results table."""
    import pandas as pd
    
    df = pd.DataFrame(results_dict).T
    
    if save_path:
        df.to_csv(save_path)
        df.to_latex(save_path.replace('.csv', '.tex'), float_format='%.4f')
    
    return df
```

---

## Reproducibility Requirements

### Random Seed Management

**All experiments must use fixed seeds**:
- **Global seed**: 42 (default)
- **Dataset splitting seed**: 42
- **Model initialization seed**: 42
- **Data augmentation seed**: Set per epoch

**Implementation**:
```python
from src.utils.reproducibility import set_seed

set_seed(42, deterministic=True)  # For exact reproduction
```

### Version Pinning

**Exact versions** (in `requirements.txt`):
```
torch==2.0.0
torchvision==0.15.0
numpy==1.24.0
scikit-learn==1.2.0
```

### Environment Configuration

**Hardware Requirements**:
- NVIDIA GPU with CUDA 11.8+
- 16GB RAM minimum
- 50GB storage

**Software Environment**:
- Ubuntu 20.04+ or compatible
- Python 3.8-3.10
- CUDA 11.8
- cuDNN 8.6+

### Expected Output Structure

```
results/
├── cifar10_6known/
│   ├── metrics.json          # All numeric results
│   ├── roc_curve.png         # ROC curve visualization
│   ├── pr_curve.png          # Precision-recall curve
│   ├── score_dist.png        # Score distribution
│   ├── checkpoints/          # Model checkpoints
│   │   ├── best_model.pth
│   │   └── final_model.pth
│   └── logs/
│       ├── training.log
│       └── evaluation.log
```

### Reproducibility Checklist

✅ Fixed random seeds set before all operations
✅ Deterministic CUDA operations enabled (if exact reproduction needed)
✅ Dataset splits saved and loaded consistently
✅ Exact hyperparameter configuration saved
✅ Model architecture specifications documented
✅ Training curves and checkpoints saved
✅ Evaluation metrics computed consistently
✅ Environment specifications recorded

---

## Troubleshooting Guide

### Issue 1: Performance Below Expected

**Symptoms**: AUROC < expected by >5%

**Potential Causes**:
1. Incorrect normalization
2. Background class not included
3. Insufficient training epochs
4. Wrong learning rate

**Debugging Steps**:
1. Verify data preprocessing matches paper
2. Check training loss convergence
3. Visualize reconstructions (should be clear for known, poor for unknown)
4. Ensure background ratio is 30%

### Issue 2: Training Instability

**Symptoms**: Loss spikes, NaN values

**Solutions**:
1. Reduce learning rate (try 0.0005)
2. Add gradient clipping (max_norm=1.0)
3. Check for batch norm issues (ensure drop_last=True)
4. Verify data normalization

### Issue 3: Memory Issues

**Symptoms**: CUDA out of memory

**Solutions**:
1. Reduce batch size (try 64 or 32)
2. Use gradient accumulation
3. Reduce model size (use custom encoder instead of ResNet)
4. Enable mixed precision training

### Issue 4: Slow Training

**Optimization Strategies**:
1. Increase num_workers for data loading
2. Enable pin_memory=True
3. Use DataLoader prefetch
4. Profile bottlenecks with PyTorch profiler
