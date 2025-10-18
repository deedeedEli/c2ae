# RISKS_AND_NOTES.md - Implementation Challenges and Solutions

## Paper Analysis Summary

**Paper Title**: C2AE: Class Conditioned Auto-Encoder for Open-Set Recognition (CVPR 2019)
**Authors**: Poojan Oza and Vishal M. Patel

**Methodology Clarity Level**: 7/10

**Strengths**:
- Clear high-level concept and motivation
- Well-defined experimental protocols with specific datasets
- Comprehensive comparison with baselines
- Good ablation studies

**Weaknesses**:
- Limited architectural details for encoder/decoder
- FiLM layer integration points not explicitly specified
- Background class generation strategy underspecified
- Some hyperparameters mentioned only in supplementary materials
- Threshold selection method not clearly defined

**Missing Implementation Details Identified**:
1. Exact encoder architecture (mentions "ResNet-based" but no specifics)
2. Number and positions of FiLM layers in decoder
3. Class embedding initialization strategy
4. Background augmentation specific techniques
5. Loss function beyond basic reconstruction (any regularization?)
6. Training convergence criteria
7. Optimal threshold selection method
8. Batch composition strategy (ratio of background samples)

**Assumptions Made in Interpretation**:
- Encoder uses standard ResNet18 without final classification layer
- FiLM applied at each decoder upsampling block
- Class embeddings initialized with small random values
- Background uses strong augmentation on unknown classes
- Standard MSE reconstruction loss without additional terms
- Training for fixed 100 epochs with step LR decay
- Threshold selected by maximizing F1 on validation set

---

## Technical Risk Assessment

### HIGH PRIORITY RISKS

#### Risk H1: FiLM Layer Implementation Ambiguity

**Issue**: Paper references FiLM (Feature-wise Linear Modulation) from Perez et al. 2017 but doesn't specify exact integration points in decoder.

**Impact**: Could significantly affect model's conditioning capability and reconstruction quality.

**Proposed Solution**:
```python
# Apply FiLM after each upsampling layer in decoder
# Position 0: After first upsample (512->256 channels)
# Position 1: After second upsample (256->128 channels)
# Position 2: After third upsample (128->64 channels)
# Position 3: After fourth upsample (64->32 channels)

film_positions = [0, 1, 2, 3]  # All decoder layers
```

**Rationale**: More conditioning points should improve class-specific reconstruction. Paper's ablation study shows performance improves with more FiLM layers.

**Fallback**: If memory/performance issues arise, reduce to positions [1, 2] (middle layers only).

**Validation**: 
1. Check that reconstruction quality differs visibly between classes
2. Verify gradient flow through FiLM layers
3. Compare ablation results with paper's Table 3

**CLAUDE CODE TASK**: In `src/models/decoder.py`, implement FiLM at positions specified in config, with default [0,1,2,3]. Add validation in forward pass to ensure class embeddings properly modulate features.

---

#### Risk H2: Background Class Generation Strategy

**Issue**: Paper states "background class with aggressive data augmentation" but doesn't specify which augmentations or their intensities.

**Impact**: Background class quality directly affects unknown detection capability. Too weak = model overfits; too strong = background doesn't represent unknown distribution.

**Proposed Solution**:
```python
# Strong augmentation pipeline
augmentations = [
    RandomCrop(32, padding=8),           # More aggressive than training
    RandomHorizontalFlip(p=0.5),
    RandomVerticalFlip(p=0.3),           # Added vertical flip
    RandomRotation(degrees=30),          # Larger rotation range
    ColorJitter(
        brightness=0.5,                   # Stronger than normal 0.2
        contrast=0.5,
        saturation=0.5,
        hue=0.3
    ),
    RandomGrayscale(p=0.3),              # Occasional grayscale
    GaussianBlur(kernel_size=3, p=0.3),  # Blur some samples
]

# Sample from unknown classes (not seen in training)
# Apply strong augmentation to create OOD samples
```

**Assumption 1**: Background samples are drawn from unknown classes with augmentation strength ~2-3x normal training augmentation.

**Rationale**: Stronger augmentation creates more diverse OOD examples, helping model learn class-conditional boundaries better.

**Fallback Options**:
1. **Option A**: Use only unknown class samples without augmentation
2. **Option B**: Use Gaussian noise as background
3. **Option C**: Mix of unknown classes + random patches

**Validation**:
1. Visualize background samples - should look "weird but plausible"
2. Check reconstruction errors: background should have errors between known and completely random
3. Ablation: Try different augmentation strengths (0.5, 0.8, 1.0)

**Experiment**: Run grid search over background ratios (0.1, 0.3, 0.5) and augmentation strengths (0.5, 0.8, 1.0).

**CLAUDE CODE TASK**: In `src/data/background_generator.py`, implement configurable augmentation strength parameter. Create ablation experiment in `experiments/ablation_background.py` to test different strategies.

---

#### Risk H3: Reconstruction Error Aggregation for Multi-Class Scoring

**Issue**: For open-set prediction, must try reconstructing with each known class and aggregate errors. Paper doesn't specify aggregation method clearly.

**Impact**: Wrong aggregation could fail to detect unknowns or misclassify knowns.

**Proposed Solution**:
```python
# For each test sample:
# 1. Try reconstructing with each known class embedding
# 2. Compute reconstruction error for each
# 3. Take MINIMUM error as the openness score

def predict_openset(x, known_classes):
    errors = []
    for class_idx in known_classes:
        reconstructed = model(x, class_idx)
        error = mse(x, reconstructed)
        errors.append(error)
    
    # Minimum error = best fitting class
    min_error = min(errors)
    predicted_class = known_classes[argmin(errors)]
    
    # If min_error > threshold, classify as unknown
    is_unknown = (min_error > threshold)
    return predicted_class, is_unknown, min_error
```

**Assumption 2**: Use minimum reconstruction error across all known classes as the openness score. Lower error = more likely to be known class.

**Rationale**: A known class sample should reconstruct well with its true class embedding. Unknown samples won't reconstruct well with any known embedding.

**Alternative Aggregations to Consider**:
- **Mean error**: Average across all classes (less discriminative)
- **Max error**: Maximum error (too sensitive to outliers)
- **Top-2 difference**: Difference between best and second-best (could help with confidence)

**Validation**:
1. Known class samples should have low minimum errors
2. Unknown samples should have high minimum errors
3. Gap between known and unknown distributions should be large

**CLAUDE CODE TASK**: Implement in `src/models/c2ae.py` with configurable aggregation method. Default to 'min' but allow 'mean', 'max' via config parameter.

---

#### Risk H4: Training Loss Specification

**Issue**: Paper mentions "reconstruction loss" but doesn't specify exact formulation or any additional regularization terms.

**Impact**: Different loss functions could lead to different learned representations.

**Proposed Solution**:
```python
# Primary reconstruction loss: MSE
reconstruction_loss = F.mse_loss(reconstructed, original, reduction='mean')

# Total loss (initially just reconstruction)
total_loss = reconstruction_loss

# Optional: Add perceptual loss for better image quality (not in paper)
# perceptual_loss = perceptual_criterion(reconstructed, original)
# total_loss = reconstruction_loss + 0.1 * perceptual_loss
```

**Assumption 3**: Use pixel-wise MSE as the only loss function, averaged over batch.

**Rationale**: Paper doesn't mention additional losses; keeping it simple aligns with autoencoder literature.

**Potential Enhancements** (if basic approach doesn't work):
1. **Perceptual loss**: Use VGG features for better image quality
2. **L1 loss**: More robust to outliers than MSE
3. **SSIM loss**: Structure-preserving reconstruction
4. **Embedding regularization**: Prevent class embeddings from collapsing

**Warning**: Don't add enhancements unless necessary - could diverge from paper.

**Validation**:
1. Reconstruction loss should decrease steadily
2. Reconstructed images should be visually recognizable
3. Known class reconstructions should be sharper than unknown

**CLAUDE CODE TASK**: Implement in `src/training/losses.py` starting with pure MSE. Add optional perceptual loss as commented-out code for troubleshooting.

---

#### Risk H5: Threshold Selection Method

**Issue**: Paper doesn't clearly specify how to select the decision threshold for unknown detection.

**Impact**: Wrong threshold severely impacts open-set performance metrics.

**Proposed Solution**:
```python
# Method 1: Maximize F1 score on validation set
def select_optimal_threshold(val_known_scores, val_unknown_scores):
    thresholds = np.percentile(
        np.concatenate([val_known_scores, val_unknown_scores]),
        np.linspace(0, 100, 1000)
    )
    
    best_f1 = 0
    best_threshold = 0
    
    for threshold in thresholds:
        # Classify based on threshold
        known_preds = (val_known_scores < threshold)
        unknown_preds = (val_unknown_scores >= threshold)
        
        # Compute F1
        tp = unknown_preds.sum()
        fp = (~known_preds).sum()
        fn = (~unknown_preds).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold

# Method 2: Fixed percentile of validation known scores
threshold = np.percentile(val_known_scores, 95)  # 95th percentile

# Method 3: Mean + k*std of known scores
threshold = val_known_scores.mean() + 2 * val_known_scores.std()
```

**Assumption 4**: Use F1-maximization on validation set as primary threshold selection method.

**Rationale**: F1 balances precision and recall, suitable for open-set scenarios.

**Fallback**: If validation set is too small or unrepresentative, use 95th percentile of known scores.

**Validation**: Selected threshold should achieve FPR ≈ 5% on validation set.

**CLAUDE CODE TASK**: Implement all three methods in `src/evaluation/threshold_selection.py`. Default to F1-maximization with fallback to percentile method.

---

### MEDIUM PRIORITY RISKS

#### Risk M1: Encoder Architecture Choice

**Issue**: Paper mentions "ResNet-based encoder" but actual architecture could be ResNet18, ResNet34, or custom variant.

**Impact**: Model capacity affects representation quality and training time.

**Proposed Solution**: Start with ResNet18 (lighter, faster), upgrade to ResNet34 if performance is below paper.

**Assumption 5**: ResNet18 pre-activation blocks without final FC layer, followed by projection to latent dimension.

**Validation**: Check if model has ~11M parameters (approximately what paper suggests).

**CLAUDE CODE TASK**: Implement configurable architecture in `src/models/encoder.py` with options ['resnet18', 'resnet34', 'custom']. Default to 'custom' for CIFAR (smaller images don't need full ResNet).

---

#### Risk M2: Embedding Dimension Selection

**Issue**: Paper uses embedding dimension but doesn't justify the choice.

**Impact**: Too small = insufficient conditioning; too large = overfitting.

**Proposed Solution**: Use 128 as default (common in literature), test 64 and 256 in ablation.

**Assumption 6**: Embedding dimension of 128 provides sufficient capacity without overfitting.

**Validation**: Embeddings should form distinct clusters for different classes in t-SNE visualization.

**CLAUDE CODE TASK**: Add t-SNE visualization of class embeddings in `src/evaluation/visualization.py`.

---

#### Risk M3: Learning Rate and Optimizer Settings

**Issue**: Paper mentions Adam optimizer but learning rate schedule details are sparse.

**Impact**: Wrong LR causes slow convergence or instability.

**Proposed Solution**:
```yaml
optimizer: adam
learning_rate: 0.001
weight_decay: 0.0001
scheduler: step
step_size: 30  # Decay every 30 epochs
gamma: 0.1      # Multiply LR by 0.1
```

**Assumption 7**: Standard Adam with step LR decay (similar to image classification literature).

**Fallback**: If training doesn't converge, try:
1. Lower LR: 0.0005
2. Cosine annealing scheduler
3. Warm-up for first 5 epochs

**Validation**: Training loss should decrease smoothly without oscillations.

**CLAUDE CODE TASK**: Implement in `src/training/scheduler.py` with multiple scheduler options.

---

#### Risk M4: Batch Normalization in Decoder

**Issue**: Decoder uses batch normalization, which can be tricky with conditioning.

**Impact**: BN statistics might interfere with class conditioning.

**Proposed Solution**: Use BatchNorm2d after upsampling but before FiLM layer:
```python
upsample -> BatchNorm -> ReLU -> FiLM
```

**Assumption 8**: Apply BN before FiLM so that FiLM modulates normalized features.

**Alternative**: Use Instance Normalization or Layer Normalization instead of Batch Normalization.

**Validation**: Check that FiLM parameters (gamma, beta) have reasonable magnitudes (~1.0 for gamma, ~0 for beta).

**CLAUDE CODE TASK**: Implement as specified. Add logging to track FiLM parameter statistics during training.

---

#### Risk M5: Dataset-Specific Preprocessing

**Issue**: Different datasets may need different normalization and augmentation.

**Impact**: Improper preprocessing degrades performance.

**Proposed Solution**: Dataset-specific configurations:
```yaml
# mnist.yaml
augmentation:
  train:
    random_crop: false        # MNIST doesn't benefit from crop
    random_horizontal_flip: false
    color_jitter: false       # Grayscale image
    normalize: true
    normalization_mean: [0.5]
    normalization_std: [0.5]

# cifar10.yaml
augmentation:
  train:
    random_crop: true
    crop_padding: 4
    random_horizontal_flip: true
    color_jitter: true        # Color images benefit
    normalize: true
    normalization_mean: [0.5, 0.5, 0.5]
    normalization_std: [0.5, 0.5, 0.5]
```

**CLAUDE CODE TASK**: Create separate config files for each dataset in `configs/`.

---

#### Risk M6: Model Initialization

**Issue**: Weight initialization strategy not specified.

**Impact**: Bad initialization slows convergence or causes training failure.

**Proposed Solution**:
```python
# Encoder: Default PyTorch/torchvision initialization (Kaiming)
# Decoder: Kaiming initialization for conv layers
# FiLM generators: Initialize to identity (gamma=1, beta=0)
# Class embeddings: Small random normal (mean=0, std=0.01)

def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
```

**Assumption 9**: Use Kaiming initialization for convolutional layers, identity for FiLM.

**CLAUDE CODE TASK**: Implement initialization in model `__init__` methods.

---

### LOW PRIORITY RISKS

#### Risk L1: Gradient Clipping

**Issue**: Large gradients might cause instability.

**Solution**: Apply gradient clipping if needed:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**CLAUDE CODE TASK**: Add as optional parameter in trainer, disabled by default.

---

#### Risk L2: Mixed Precision Training

**Issue**: Training might be slow without mixed precision.

**Solution**: Use torch.cuda.amp for faster training:
```python
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    output = model(x, labels)
    loss = criterion(output, target)
```

**CLAUDE CODE TASK**: Implement as optional feature in trainer.

---

#### Risk L3: Data Loading Bottleneck

**Issue**: Data loading might be slower than GPU computation.

**Solution**: 
- Increase num_workers (4-8)
- Enable pin_memory
- Use persistent_workers=True

**CLAUDE CODE TASK**: Optimize data loader configuration in `src/data/dataset_loader.py`.

---

## Implementation Assumptions (Complete List)

### Assumption 1: Background Class Augmentation
**Assumption**: Background samples use 2-3x stronger augmentation than normal training.
**Rationale**: Creates diverse OOD examples for better boundary learning.
**Validation**: Visual inspection of background samples; ablation study.
**Fallback**: Reduce augmentation strength if model can't learn from heavily augmented data.

### Assumption 2: Minimum Error Aggregation
**Assumption**: Use minimum reconstruction error across known classes as openness score.
**Rationale**: Best-fitting class should have lowest error; unknowns won't fit any class well.
**Validation**: Clear separation in score distributions between known and unknown.
**Fallback**: Try mean or weighted aggregation.

### Assumption 3: Pure MSE Loss
**Assumption**: Use only pixel-wise MSE without additional regularization.
**Rationale**: Paper doesn't mention other losses; simpler is better for reproduction.
**Validation**: Reconstructions should be visually recognizable.
**Fallback**: Add perceptual loss if image quality is poor.

### Assumption 4: F1-Based Threshold Selection
**Assumption**: Select threshold by maximizing F1 score on validation set.
**Rationale**: Balances precision and recall for open-set detection.
**Validation**: Threshold should achieve ~5% FPR on validation set.
**Fallback**: Use 95th percentile of known scores.

### Assumption 5: ResNet18 Encoder
**Assumption**: Use ResNet18 architecture without final FC layer.
**Rationale**: Common choice in literature; appropriate capacity for CIFAR-scale images.
**Validation**: ~11M parameters total; reasonable training time.
**Fallback**: Use custom smaller encoder for MNIST/CIFAR.

### Assumption 6: Embedding Dimension 128
**Assumption**: Class embedding dimension of 128 is sufficient.
**Rationale**: Standard choice in conditioning literature; not too large to overfit.
**Validation**: Class embeddings form distinct clusters in t-SNE.
**Fallback**: Try 64 (smaller) or 256 (larger) in ablation.

### Assumption 7: Step LR Schedule
**Assumption**: Adam optimizer with step LR decay (every 30 epochs, gamma=0.1).
**Rationale**: Standard image classification schedule.
**Validation**: Smooth loss decrease; no oscillations.
**Fallback**: Cosine annealing or reduce-on-plateau.

### Assumption 8: BatchNorm Before FiLM
**Assumption**: Apply BatchNorm before FiLM layer in decoder.
**Rationale**: FiLM modulates normalized features for better conditioning.
**Validation**: FiLM parameters have reasonable magnitudes.
**Fallback**: Use Instance Normalization or Layer Normalization.

### Assumption 9: Kaiming Initialization
**Assumption**: Use Kaiming He initialization for conv layers.
**Rationale**: Best practice for ReLU networks.
**Validation**: Training converges within expected epochs.
**Fallback**: Xavier initialization or pretrained weights.

### Assumption 10: Background Ratio 30%
**Assumption**: Background class comprises 30% of training batches.
**Rationale**: Paper mentions "sufficient background samples" - 30% is common in literature.
**Validation**: Model learns to distinguish known from unknown.
**Fallback**: Try 20% or 50% in ablation study.

---

## Common Implementation Pitfalls

### Pitfall 1: Forgetting to Set Model to Eval Mode

**Issue**: BatchNorm and Dropout behave differently in train vs eval mode.

**Solution**:
```python
model.eval()
with torch.no_grad():
    # Evaluation code
```

**Detection**: Unstable evaluation metrics across runs.

---

### Pitfall 2: Not Normalizing Test Data

**Issue**: Forgetting to apply same normalization to test data as training.

**Solution**: Ensure test transforms include normalization with same stats:
```python
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
```

**Detection**: Very high reconstruction errors even for known classes.

---

### Pitfall 3: Label Mismatch Between Dataset and Model

**Issue**: Class labels might not be consecutive or zero-indexed.

**Solution**: Map labels to continuous range [0, num_known_classes]:
```python
label_map = {old_label: new_label for new_label, old_label in enumerate(known_classes)}
mapped_labels = [label_map[label] for label in original_labels]
```

**Detection**: Index out of bounds errors or assertion failures.

---

### Pitfall 4: Incorrect Batch Dimension Handling

**Issue**: Forgetting to add/remove batch dimension for single-sample inference.

**Solution**:
```python
# Single image
x = x.unsqueeze(0)  # Add batch dimension
output = model(x, label)
output = output.squeeze(0)  # Remove batch dimension
```

**Detection**: Shape mismatch errors.

---

### Pitfall 5: Not Detaching Tensors for Metric Computation

**Issue**: Computing metrics on tensors with gradients attached.

**Solution**:
```python
with torch.no_grad():
    predictions = model(x)
scores = predictions.detach().cpu().numpy()
```

**Detection**: Memory leaks; gradually increasing memory usage.

---

### Pitfall 6: Overfitting to Validation Set

**Issue**: Selecting threshold on validation set then reporting validation metrics.

**Solution**: Use separate validation and test sets, or use cross-validation:
```python
# Select threshold on validation set
threshold = select_threshold(val_known, val_unknown)

# Evaluate on test set
test_results = evaluate(test_known, test_unknown, threshold)
```

**Detection**: Unrealistically high test performance.

---

## Debugging and Validation Strategies

### Strategy 1: Sanity Checks

**Implement Progressive Testing**:

```python
# Test 1: Model can overfit to single batch
def test_overfit_single_batch():
    """Model should achieve near-zero loss on single batch."""
    model = C2AE(...)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Get single batch
    x, y = next(iter(train_loader))
    
    for _ in range(100):
        output = model(x, y)
        loss = F.mse_loss(output['reconstructed'], x)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    assert loss.item() < 0.01, f"Can't overfit single batch: loss={loss.item()}"

# Test 2: Forward pass works with different batch sizes
def test_variable_batch_size():
    """Model should handle different batch sizes."""
    model = C2AE(...)
    for bs in [1, 16, 32, 128]:
        x = torch.randn(bs, 3, 32, 32)
        y = torch.randint(0, 7, (bs,))
        output = model(x, y)
        assert output['reconstructed'].shape == x.shape

# Test 3: Reconstruction error increases for wrong class labels
def test_class_conditioning():
    """Reconstruction should be better with correct class label."""
    model = C2AE(...)
    model.eval()
    
    x, true_label = test_dataset[0]
    x = x.unsqueeze(0)
    
    with torch.no_grad():
        # Reconstruct with correct label
        correct_output = model(x, torch.tensor([true_label]))
        correct_error = F.mse_loss(x, correct_output['reconstructed'])
        
        # Reconstruct with wrong labels
        wrong_errors = []
        for wrong_label in range(num_classes):
            if wrong_label != true_label:
                wrong_output = model(x, torch.tensor([wrong_label]))
                wrong_error = F.mse_loss(x, wrong_output['reconstructed'])
                wrong_errors.append(wrong_error.item())
        
        # Correct class should have lower error
        assert correct_error.item() < np.mean(wrong_errors), \
            "Model doesn't condition on class properly"
```

**CLAUDE CODE TASK**: Implement these sanity checks in `tests/test_sanity/test_model_sanity.py`.

---

### Strategy 2: Visualization-Based Debugging

**Implement Visualization Tools**:

```python
def visualize_reconstructions(model, test_loader, num_samples=10, save_path=None):
    """Visualize original and reconstructed images."""
    model.eval()
    
    fig, axes = plt.subplots(3, num_samples, figsize=(20, 6))
    
    images, labels, _ = next(iter(test_loader))
    images = images[:num_samples]
    labels = labels[:num_samples]
    
    with torch.no_grad():
        # Reconstruct with correct labels
        correct_output = model(images, labels)
        correct_recon = correct_output['reconstructed']
        
        # Reconstruct with random wrong labels
        wrong_labels = torch.randint(0, model.num_classes, labels.shape)
        wrong_output = model(images, wrong_labels)
        wrong_recon = wrong_output['reconstructed']
    
    for i in range(num_samples):
        # Original
        axes[0, i].imshow(images[i].permute(1, 2, 0).cpu())
        axes[0, i].axis('off')
        axes[0, i].set_title(f'Original (class {labels[i]})')
        
        # Correct reconstruction
        axes[1, i].imshow(correct_recon[i].permute(1, 2, 0).cpu())
        axes[1, i].axis('off')
        axes[1, i].set_title('Correct Class')
        
        # Wrong reconstruction
        axes[2, i].imshow(wrong_recon[i].permute(1, 2, 0).cpu())
        axes[2, i].axis('off')
        axes[2, i].set_title(f'Wrong Class ({wrong_labels[i]})')
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def visualize_embedding_space(model, data_loader, save_path=None):
    """Visualize class embedding space with t-SNE."""
    from sklearn.manifold import TSNE
    
    # Get all class embeddings
    embeddings = model.class_embedding.get_all_embeddings().detach().cpu().numpy()
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=range(len(embeddings)), cmap='tab10', s=200)
    
    for i, (x, y) in enumerate(embeddings_2d):
        plt.annotate(f'Class {i}', (x, y), fontsize=12)
    
    plt.title('Class Embedding Space (t-SNE)', fontsize=14)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
```

**CLAUDE CODE TASK**: Implement in `src/evaluation/visualization.py` and call during training to monitor progress.

---

### Strategy 3: Metric Tracking

**Track Additional Metrics During Training**:

```python
def compute_training_diagnostics(model, val_loader):
    """Compute diagnostic metrics during training."""
    diagnostics = {}
    
    # 1. Reconstruction quality (PSNR)
    psnrs = []
    for images, labels, _ in val_loader:
        with torch.no_grad():
            output = model(images, labels)
            psnr = compute_psnr(images, output['reconstructed'])
            psnrs.append(psnr)
    diagnostics['mean_psnr'] = np.mean(psnrs)
    
    # 2. Embedding diversity
    embeddings = model.class_embedding.get_all_embeddings()
    # Compute pairwise distances
    distances = torch.cdist(embeddings, embeddings, p=2)
    # Average off-diagonal distance
    mask = ~torch.eye(len(embeddings), dtype=torch.bool)
    diagnostics['embedding_diversity'] = distances[mask].mean().item()
    
    # 3. FiLM parameter statistics
    for name, module in model.named_modules():
        if isinstance(module, FiLMGenerator):
            # Check gamma and beta magnitudes
            diagnostics[f'{name}_gamma_mean'] = module.gamma_network[-1].weight.mean().item()
            diagnostics[f'{name}_beta_mean'] = module.beta_network[-1].weight.mean().item()
    
    return diagnostics


def compute_psnr(original, reconstructed):
    """Compute Peak Signal-to-Noise Ratio."""
    mse = F.mse_loss(reconstructed, original)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0  # Normalized to [0, 1]
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()
```

**CLAUDE CODE TASK**: Add diagnostic computation in `src/training/trainer.py` and log every N epochs.

---

### Strategy 4: Ablation Studies

**Implement Systematic Ablation**:

```python
# Ablation 1: Number of FiLM layers
film_positions_configs = [
    [],              # No FiLM (baseline)
    [1, 2],          # Middle layers only
    [0, 1, 2, 3],    # All layers (default)
]

# Ablation 2: Background class ratio
background_ratios = [0.0, 0.1, 0.3, 0.5]

# Ablation 3: Embedding dimension
embedding_dims = [32, 64, 128, 256]

# Ablation 4: Background augmentation strength
augmentation_strengths = [0.3, 0.5, 0.8, 1.0]

# Run grid
results = {}
for film_pos in film_positions_configs:
    for bg_ratio in background_ratios:
        config = {
            'film_positions': film_pos,
            'background_ratio': bg_ratio
        }
        result = run_experiment(config)
        results[str(config)] = result
```

**CLAUDE CODE TASK**: Implement ablation experiment runner in `experiments/run_ablation_study.py`.

---

## Claude Code Specific Guidance

### Development Sequence to Minimize Backtracking

**Phase 1**: Core Infrastructure (No Dependencies)
1. Configuration system
2. Reproducibility utilities
3. Device management
4. Logging and checkpointing

**Phase 2**: Data Pipeline (Depends on Phase 1)
1. Data transforms
2. Dataset loaders
3. Open-set splitting
4. Background generation

**Phase 3**: Model Components (Depends on Phase 1)
1. FiLM layers (no dependencies)
2. Class embedding (no dependencies)
3. Encoder (no dependencies)
4. Decoder (depends on FiLM)
5. C2AE (depends on encoder, decoder, embedding)

**Phase 4**: Training (Depends on Phases 2 & 3)
1. Loss functions
2. Trainer
3. Experiment runner

**Phase 5**: Evaluation (Depends on Phase 3)
1. Metrics computation
2. Evaluator
3. Visualization

**Phase 6**: Integration and Testing (Depends on All)
1. Unit tests
2. Integration tests
3. End-to-end experiments

---

### Code Organization Patterns

**Pattern 1: Configuration-Driven Design**
```python
# Good: All parameters in config
def create_model(config):
    return C2AE(
        num_classes=config.get('model.num_classes'),
        embedding_dim=config.get('model.embedding_dim'),
        # ... all from config
    )

# Avoid: Hardcoded parameters
def create_model():
    return C2AE(num_classes=7, embedding_dim=128)  # Bad
```

**Pattern 2: Dependency Injection**
```python
# Good: Pass dependencies explicitly
class Trainer:
    def __init__(self, model, optimizer, scheduler, logger):
        self.model = model
        self.optimizer = optimizer
        # ...

# Avoid: Creating dependencies internally
class Trainer:
    def __init__(self, config):
        self.model = C2AE(...)  # Hard to test/modify
```

**Pattern 3: Return Dictionaries for Complex Outputs**
```python
# Good: Named returns
def forward(self, x, labels):
    return {
        'reconstructed': reconstructed,
        'latent': latent,
        'embedding': embedding
    }

# Avoid: Positional tuple returns
def forward(self, x, labels):
    return reconstructed, latent, embedding  # Easy to mix up
```

---

### Areas Where Human Review Might Be Needed

1. **Hyperparameter Tuning**: If initial results are >5% below paper benchmarks
2. **Architecture Modifications**: If memory constraints require smaller models
3. **Dataset-Specific Issues**: If datasets have unexpected format or corruption
4. **Numerical Instability**: If training encounters NaNs or divergence
5. **Performance Optimization**: If training is prohibitively slow (>24h per experiment)

---

## Summary Checklist for Autonomous Implementation

✅ **Configuration System**: Centralized, hierarchical, easily overridable
✅ **Data Pipeline**: Modular, tested, supports all datasets
✅ **Model Architecture**: Matches paper description with documented assumptions
✅ **Training Loop**: Stable, logged, checkpointed
✅ **Evaluation**: Comprehensive metrics matching paper
✅ **Testing**: Unit tests for all components
✅ **Visualization**: Progress monitoring and result analysis
✅ **Documentation**: Inline comments and docstrings
✅ **Reproducibility**: Seeded, deterministic, version-pinned
✅ **Error Handling**: Graceful failures with informative messages

**Estimated Total Implementation Time**: 16-20 hours
**Estimated Training Time**: 2-4 hours per dataset
**Total Time to First Results**: 20-25 hours

---

## Final Notes

**Critical Success Factors**:
1. Follow the implementation plan sequentially
2. Test each component before proceeding
3. Visualize intermediate results frequently
4. Compare with paper results at each milestone
5. Document any deviations from the plan

**If Results Don't Match Paper**:
1. First check data preprocessing and normalization
2. Verify model architecture matches assumptions
3. Review training hyperparameters
4. Check threshold selection method
5. Compare training curves with expected behavior
6. Run ablation studies to isolate issues

**When to Seek Help**:
- Training loss doesn't decrease after 10 epochs
- AUROC is >10% below paper results after full training
- Model produces NaN or exploding gradients
- Systematic failure across all datasets
- Reproducibility issues (same config, different results)

Good luck with the implementation! 🚀
