import torch
import numpy as np
from typing import Dict, List
from tqdm import tqdm
from .metrics import (
    compute_auroc, compute_aupr, compute_fpr_at_tpr,
    compute_ccr_at_fpr, compute_optimal_threshold,
    compute_classification_accuracy
)


class OpenSetEvaluator:
    """
    Evaluator for open-set recognition performance.
    
    Args:
        model: Trained C2AE model
        known_classes: List of known class indices
        device: Evaluation device
    """
    
    def __init__(self, model, known_classes: List[int], device: torch.device):
        self.model = model
        self.known_classes = known_classes
        self.device = device
        self.model.eval()
    
    def compute_openset_scores(self, data_loader) -> tuple:
        """
        Compute reconstruction-based openness scores.
        
        Args:
            data_loader: Data to score
        
        Returns:
            tuple: (scores, labels, predictions)
                - scores: np.ndarray, minimum reconstruction errors
                - labels: np.ndarray, true labels
                - predictions: np.ndarray, predicted class labels
        """
        all_scores = []
        all_labels = []
        all_predictions = []
        
        with torch.no_grad():
            for images, labels, _ in tqdm(data_loader, desc="Computing scores"):
                images = images.to(self.device)
                batch_size = images.size(0)
                
                # Compute reconstruction error for each known class
                reconstruction_errors = torch.zeros(batch_size, len(self.known_classes), device=self.device)
                
                for i, class_idx in enumerate(self.known_classes):
                    class_labels = torch.full((batch_size,), class_idx, dtype=torch.long, device=self.device)
                    outputs = self.model(images, class_labels)
                    errors = self.model.compute_reconstruction_error(
                        images, outputs['reconstructed'], reduction='mean'
                    )
                    reconstruction_errors[:, i] = errors
                
                # Use minimum reconstruction error as score
                min_errors, best_class_indices = reconstruction_errors.min(dim=1)
                
                # Map to actual class labels
                predictions = torch.tensor(
                    [self.known_classes[i] for i in best_class_indices.cpu().numpy()],
                    dtype=torch.long
                )
                
                all_scores.append(min_errors.cpu().numpy())
                all_labels.append(labels.numpy())
                all_predictions.append(predictions.numpy())
        
        scores = np.concatenate(all_scores)
        labels = np.concatenate(all_labels)
        predictions = np.concatenate(all_predictions)
        
        return scores, labels, predictions
    
    def evaluate(self, test_known_loader, test_unknown_loader) -> Dict[str, float]:
        """
        Comprehensive open-set evaluation.
        
        Args:
            test_known_loader: Known class test data
            test_unknown_loader: Unknown class test data
        
        Returns:
            Dictionary with evaluation metrics
        """
        print("Evaluating on known classes...")
        known_scores, known_labels, known_predictions = self.compute_openset_scores(test_known_loader)
        
        print("Evaluating on unknown classes...")
        unknown_scores, unknown_labels, _ = self.compute_openset_scores(test_unknown_loader)
        
        # Compute metrics
        results = {}
        
        # AUROC
        results['auroc'] = compute_auroc(known_scores, unknown_scores)
        print(f"AUROC: {results['auroc']:.4f}")
        
        # AUPR
        results['aupr'] = compute_aupr(known_scores, unknown_scores)
        print(f"AUPR: {results['aupr']:.4f}")
        
        # FPR at 95% TPR
        results['fpr_at_95_tpr'] = compute_fpr_at_tpr(known_scores, unknown_scores, tpr_threshold=0.95)
        print(f"FPR at 95% TPR: {results['fpr_at_95_tpr']:.4f}")
        
        # CCR at different FPR thresholds
        for fpr_thresh in [0.01, 0.05, 0.1]:
            ccr_result = compute_ccr_at_fpr(
                known_scores, known_predictions, known_labels,
                unknown_scores, fpr_threshold=fpr_thresh
            )
            results[f'ccr_at_fpr_{int(fpr_thresh*100)}'] = ccr_result
            print(f"CCR at FPR={fpr_thresh}: {ccr_result['ccr']:.4f} (threshold={ccr_result['threshold']:.4f})")
        
        # Optimal threshold
        results['optimal_threshold'] = compute_optimal_threshold(known_scores, unknown_scores, method='f1')
        print(f"Optimal threshold (F1): {results['optimal_threshold']:.4f}")
        
        # Classification accuracy on known classes
        accuracy_results = compute_classification_accuracy(
            known_predictions, known_labels, num_classes=len(self.known_classes)
        )
        results['known_classification_accuracy'] = accuracy_results['overall_accuracy']
        results['per_class_accuracy'] = accuracy_results['per_class_accuracy']
        print(f"Known class accuracy: {results['known_classification_accuracy']:.4f}")
        
        return results
