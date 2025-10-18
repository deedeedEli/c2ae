import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve
import seaborn as sns
from pathlib import Path


def plot_roc_curve(known_scores, unknown_scores, save_path=None):
    """
    Plot ROC curve for open-set detection.
    
    Args:
        known_scores: Reconstruction errors for known classes
        unknown_scores: Reconstruction errors for unknown classes
        save_path: Path to save the figure
    """
    all_scores = np.concatenate([known_scores, unknown_scores])
    labels = np.concatenate([np.zeros(len(known_scores)), np.ones(len(unknown_scores))])
    
    fpr, tpr, _ = roc_curve(labels, all_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label='ROC Curve')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve for Open-Set Detection', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_precision_recall_curve(known_scores, unknown_scores, save_path=None):
    """
    Plot Precision-Recall curve.
    
    Args:
        known_scores: Reconstruction errors for known classes
        unknown_scores: Reconstruction errors for unknown classes
        save_path: Path to save the figure
    """
    all_scores = np.concatenate([known_scores, unknown_scores])
    labels = np.concatenate([np.zeros(len(known_scores)), np.ones(len(unknown_scores))])
    
    precision, recall, _ = precision_recall_curve(labels, all_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, label='PR Curve')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_score_distributions(known_scores, unknown_scores, save_path=None):
    """
    Plot distribution of reconstruction scores.
    
    Args:
        known_scores: Reconstruction errors for known classes
        unknown_scores: Reconstruction errors for unknown classes
        save_path: Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    
    plt.hist(known_scores, bins=50, alpha=0.6, label='Known Classes', density=True)
    plt.hist(unknown_scores, bins=50, alpha=0.6, label='Unknown Classes', density=True)
    
    plt.xlabel('Reconstruction Error', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Distribution of Reconstruction Errors', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_training_history(history, save_path=None):
    """
    Plot training history.
    
    Args:
        history: Dictionary with 'train_loss' and 'val_loss' lists
        save_path: Path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    # Learning rate
    ax2.plot(epochs, history['learning_rate'], linewidth=2, color='green')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Learning Rate', fontsize=12)
    ax2.set_title('Learning Rate Schedule', fontsize=14)
    ax2.set_yscale('log')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(predictions, labels, class_names, save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        predictions: Predicted labels
        labels: True labels
        class_names: List of class names
        save_path: Path to save the figure
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(labels, predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_evaluation_report(results, save_dir='./results/visualizations'):
    """
    Create comprehensive evaluation report with visualizations.
    
    Args:
        results: Dictionary with evaluation results
        save_dir: Directory to save visualizations
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating evaluation report in {save_dir}...")
    
    # Create visualizations if score data is available
    if 'known_scores' in results and 'unknown_scores' in results:
        plot_roc_curve(
            results['known_scores'],
            results['unknown_scores'],
            save_path=save_dir / 'roc_curve.png'
        )
        
        plot_precision_recall_curve(
            results['known_scores'],
            results['unknown_scores'],
            save_path=save_dir / 'precision_recall_curve.png'
        )
        
        plot_score_distributions(
            results['known_scores'],
            results['unknown_scores'],
            save_path=save_dir / 'score_distributions.png'
        )
    
    print(f"Evaluation report saved to {save_dir}")
