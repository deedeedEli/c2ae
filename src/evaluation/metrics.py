import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from typing import Tuple, Dict


def compute_auroc(known_scores: np.ndarray, unknown_scores: np.ndarray) -> float:
    """
    Compute Area Under ROC Curve.
    
    Args:
        known_scores: Reconstruction errors for known classes (lower is better)
        unknown_scores: Reconstruction errors for unknown classes (higher expected)
    
    Returns:
        AUROC value in [0, 1]
    """
    # Combine scores and create labels
    # Known = 0 (inlier), Unknown = 1 (outlier)
    all_scores = np.concatenate([known_scores, unknown_scores])
    labels = np.concatenate([np.zeros(len(known_scores)), np.ones(len(unknown_scores))])
    
    # Higher reconstruction error should indicate unknown
    # So we use scores directly (higher score = more likely unknown)
    auroc = roc_auc_score(labels, all_scores)
    
    return auroc


def compute_aupr(known_scores: np.ndarray, unknown_scores: np.ndarray) -> float:
    """
    Compute Area Under Precision-Recall Curve.
    
    Args:
        known_scores: Scores for known classes
        unknown_scores: Scores for unknown classes
    
    Returns:
        AUPR value in [0, 1]
    """
    all_scores = np.concatenate([known_scores, unknown_scores])
    labels = np.concatenate([np.zeros(len(known_scores)), np.ones(len(unknown_scores))])
    
    aupr = average_precision_score(labels, all_scores)
    
    return aupr


def compute_fpr_at_tpr(known_scores: np.ndarray, 
                       unknown_scores: np.ndarray, 
                       tpr_threshold: float = 0.95) -> float:
    """
    Compute False Positive Rate at a given True Positive Rate.
    
    Args:
        known_scores: Scores for known classes
        unknown_scores: Scores for unknown classes
        tpr_threshold: Desired TPR (e.g., 0.95)
    
    Returns:
        FPR at the specified TPR
    """
    all_scores = np.concatenate([known_scores, unknown_scores])
    labels = np.concatenate([np.zeros(len(known_scores)), np.ones(len(unknown_scores))])
    
    fpr, tpr, thresholds = roc_curve(labels, all_scores)
    
    # Find FPR at desired TPR
    idx = np.argmin(np.abs(tpr - tpr_threshold))
    fpr_at_tpr = fpr[idx]
    
    return fpr_at_tpr


def compute_ccr_at_fpr(known_scores: np.ndarray,
                       known_predictions: np.ndarray,
                       known_labels: np.ndarray,
                       unknown_scores: np.ndarray,
                       fpr_threshold: float = 0.05) -> Dict[str, float]:
    """
    Compute Correct Classification Rate at a given False Positive Rate.
    
    Args:
        known_scores: Reconstruction errors for known samples
        known_predictions: Predicted classes for known samples
        known_labels: True labels for known samples
        unknown_scores: Reconstruction errors for unknown samples
        fpr_threshold: Desired FPR (e.g., 0.05)
    
    Returns:
        Dictionary with CCR and threshold
    """
    all_scores = np.concatenate([known_scores, unknown_scores])
    labels = np.concatenate([np.zeros(len(known_scores)), np.ones(len(unknown_scores))])
    
    fpr, tpr, thresholds = roc_curve(labels, all_scores)
    
    # Find threshold at desired FPR
    idx = np.argmin(np.abs(fpr - fpr_threshold))
    threshold = thresholds[idx]
    actual_fpr = fpr[idx]
    
    # Compute CCR: correct classifications among samples below threshold
    known_below_threshold = known_scores < threshold
    if np.sum(known_below_threshold) > 0:
        correct = known_predictions[known_below_threshold] == known_labels[known_below_threshold]
        ccr = np.mean(correct)
    else:
        ccr = 0.0
    
    return {
        'ccr': ccr,
        'threshold': threshold,
        'actual_fpr': actual_fpr
    }


def compute_optimal_threshold(known_scores: np.ndarray, 
                             unknown_scores: np.ndarray,
                             method: str = 'f1') -> float:
    """
    Compute optimal threshold for unknown detection.
    
    Args:
        known_scores: Scores for known classes
        unknown_scores: Scores for unknown classes
        method: Method to use ('f1', 'youden', 'precision_recall')
    
    Returns:
        Optimal threshold value
    """
    all_scores = np.concatenate([known_scores, unknown_scores])
    labels = np.concatenate([np.zeros(len(known_scores)), np.ones(len(unknown_scores))])
    
    if method == 'f1':
        # Maximize F1 score
        precision, recall, thresholds = precision_recall_curve(labels, all_scores)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_idx = np.argmax(f1_scores[:-1])  # Exclude last element (threshold at infinity)
        optimal_threshold = thresholds[optimal_idx]
        
    elif method == 'youden':
        # Maximize Youden's J statistic (TPR - FPR)
        fpr, tpr, thresholds = roc_curve(labels, all_scores)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return optimal_threshold


def compute_classification_accuracy(predictions: np.ndarray, 
                                   labels: np.ndarray,
                                   num_classes: int) -> Dict[str, float]:
    """
    Compute per-class and overall accuracy.
    
    Args:
        predictions: Predicted class labels
        labels: True class labels
        num_classes: Number of classes
    
    Returns:
        Dictionary with overall and per-class accuracy
    """
    overall_accuracy = np.mean(predictions == labels)
    
    per_class_accuracy = {}
    for class_idx in range(num_classes):
        mask = labels == class_idx
        if np.sum(mask) > 0:
            per_class_accuracy[class_idx] = np.mean(predictions[mask] == labels[mask])
        else:
            per_class_accuracy[class_idx] = 0.0
    
    return {
        'overall_accuracy': overall_accuracy,
        'per_class_accuracy': per_class_accuracy
    }
