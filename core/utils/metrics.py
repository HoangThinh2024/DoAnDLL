import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, top_k_accuracy_score
)
from typing import List, Dict, Any
import torch


class MetricsCalculator:
    """Calculate various metrics for multiclass classification"""
    
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
    
    def calculate_metrics(self, y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        """Calculate comprehensive metrics"""
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Precision, Recall, F1 - macro average
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Precision, Recall, F1 - weighted average
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        return metrics
    
    def calculate_top_k_accuracy(self, y_true: List[int], y_pred_proba: np.ndarray, k: int = 5) -> float:
        """Calculate top-k accuracy"""
        return top_k_accuracy_score(y_true, y_pred_proba, k=k)
    
    def get_confusion_matrix(self, y_true: List[int], y_pred: List[int]) -> np.ndarray:
        """Get confusion matrix"""
        return confusion_matrix(y_true, y_pred)
    
    def get_classification_report(self, y_true: List[int], y_pred: List[int], 
                                target_names: List[str] = None) -> str:
        """Get detailed classification report"""
        return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    
    def calculate_per_class_metrics(self, y_true: List[int], y_pred: List[int]) -> Dict[int, Dict[str, float]]:
        """Calculate per-class metrics"""
        
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        per_class_metrics = {}
        
        # Calculate metrics for each class
        for class_id in range(self.num_classes):
            class_mask = np.array(y_true) == class_id
            if np.sum(class_mask) == 0:  # Skip if no samples for this class
                continue
            
            class_pred = np.array(y_pred)[class_mask]
            class_true = np.array(y_true)[class_mask]
            
            # Binary classification metrics for this class
            class_acc = accuracy_score(class_true, class_pred)
            class_prec = precision_score(class_true == class_id, class_pred == class_id, zero_division=0)
            class_rec = recall_score(class_true == class_id, class_pred == class_id, zero_division=0)
            class_f1 = f1_score(class_true == class_id, class_pred == class_id, zero_division=0)
            
            per_class_metrics[class_id] = {
                'accuracy': class_acc,
                'precision': class_prec,
                'recall': class_rec,
                'f1_score': class_f1,
                'support': np.sum(class_mask)
            }
        
        return per_class_metrics
    
    def calculate_confusion_matrix_metrics(self, confusion_matrix):
        """Calculate additional metrics from confusion matrix"""
        # True positives, false positives, false negatives for each class
        tp = np.diag(confusion_matrix)
        fp = np.sum(confusion_matrix, axis=0) - tp
        fn = np.sum(confusion_matrix, axis=1) - tp
        tn = np.sum(confusion_matrix) - (fp + fn + tp)
        
        # Calculate sensitivity (recall) and specificity for each class
        sensitivity = tp / (tp + fn + 1e-8)  # Avoid division by zero
        specificity = tn / (tn + fp + 1e-8)
        
        return {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'true_negatives': tn
        }
    
    def calculate_advanced_metrics(self, y_true: List[int], y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calculate advanced metrics including AUC, entropy, etc."""
        from sklearn.metrics import roc_auc_score, log_loss
        
        metrics = {}
        
        try:
            # Multi-class AUC (one-vs-rest)
            metrics['auc_ovr'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
            
            # Multi-class AUC (one-vs-one)
            metrics['auc_ovo'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovo')
        except ValueError:
            # Handle cases where not all classes are present
            metrics['auc_ovr'] = 0.0
            metrics['auc_ovo'] = 0.0
        
        # Cross-entropy loss
        try:
            metrics['cross_entropy'] = log_loss(y_true, y_pred_proba)
        except ValueError:
            metrics['cross_entropy'] = float('inf')
        
        # Prediction confidence statistics
        max_probs = np.max(y_pred_proba, axis=1)
        metrics['mean_confidence'] = np.mean(max_probs)
        metrics['std_confidence'] = np.std(max_probs)
        metrics['min_confidence'] = np.min(max_probs)
        metrics['max_confidence'] = np.max(max_probs)
        
        # Entropy of predictions
        entropy = -np.sum(y_pred_proba * np.log(y_pred_proba + 1e-8), axis=1)
        metrics['mean_entropy'] = np.mean(entropy)
        metrics['std_entropy'] = np.std(entropy)
        
        return metrics
    
    def create_classification_report_dict(self, y_true: List[int], y_pred: List[int], 
                                        class_names: List[str] = None) -> Dict[str, Any]:
        """Create a comprehensive classification report as dictionary"""
        from sklearn.metrics import classification_report
        
        # Get basic classification report
        if class_names:
            target_names = class_names
        else:
            target_names = [f'Class_{i}' for i in range(self.num_classes)]
        
        report = classification_report(
            y_true, y_pred, 
            target_names=target_names, 
            output_dict=True, 
            zero_division=0
        )
        
        # Add confusion matrix
        cm = self.get_confusion_matrix(y_true, y_pred)
        report['confusion_matrix'] = cm.tolist()
        
        # Add per-class metrics
        per_class = self.calculate_per_class_metrics(y_true, y_pred)
        report['per_class_detailed'] = per_class
        
        return report
    
    def calculate_class_balance_metrics(self, y_true: List[int]) -> Dict[str, Any]:
        """Calculate metrics related to class balance"""
        unique, counts = np.unique(y_true, return_counts=True)
        total_samples = len(y_true)
        
        # Class distribution
        class_distribution = {int(cls): int(count) for cls, count in zip(unique, counts)}
        class_percentages = {int(cls): count/total_samples*100 for cls, count in zip(unique, counts)}
        
        # Balance metrics
        max_count = np.max(counts)
        min_count = np.min(counts)
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        # Gini coefficient for class imbalance
        sorted_counts = np.sort(counts)
        n = len(sorted_counts)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_counts)) / (n * np.sum(sorted_counts)) - (n + 1) / n
        
        return {
            'class_distribution': class_distribution,
            'class_percentages': class_percentages,
            'imbalance_ratio': imbalance_ratio,
            'gini_coefficient': gini,
            'num_classes_present': len(unique),
            'most_frequent_class': int(unique[np.argmax(counts)]),
            'least_frequent_class': int(unique[np.argmin(counts)])
        }
    
    def plot_confusion_matrix(self, y_true: List[int], y_pred: List[int], 
                            class_names: List[str] = None, normalize: bool = False):
        """Plot confusion matrix using matplotlib"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            cm = self.get_confusion_matrix(y_true, y_pred)
            
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                fmt = '.2f'
            else:
                fmt = 'd'
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names)
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            
            return plt.gcf()
        except ImportError:
            print("Matplotlib/Seaborn not available for plotting")
            return None
    
    def calculate_calibration_metrics(self, y_true: List[int], y_pred_proba: np.ndarray, 
                                    n_bins: int = 10) -> Dict[str, Any]:
        """Calculate calibration metrics for probability predictions"""
        try:
            from sklearn.calibration import calibration_curve
            
            # For multi-class, we'll calculate calibration for each class
            calibration_results = {}
            
            for class_idx in range(y_pred_proba.shape[1]):
                # Binary classification: class vs rest
                y_binary = (np.array(y_true) == class_idx).astype(int)
                prob_class = y_pred_proba[:, class_idx]
                
                if len(np.unique(y_binary)) == 2:  # Only if both classes present
                    fraction_pos, mean_pred_value = calibration_curve(
                        y_binary, prob_class, n_bins=n_bins
                    )
                    
                    # Expected Calibration Error (ECE)
                    bin_boundaries = np.linspace(0, 1, n_bins + 1)
                    bin_lowers = bin_boundaries[:-1]
                    bin_uppers = bin_boundaries[1:]
                    
                    ece = 0
                    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                        in_bin = (prob_class > bin_lower) & (prob_class <= bin_upper)
                        prop_in_bin = in_bin.mean()
                        
                        if prop_in_bin > 0:
                            accuracy_in_bin = y_binary[in_bin].mean()
                            avg_confidence_in_bin = prob_class[in_bin].mean()
                            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                    
                    calibration_results[f'class_{class_idx}'] = {
                        'ece': ece,
                        'fraction_positives': fraction_pos.tolist(),
                        'mean_predicted_value': mean_pred_value.tolist()
                    }
            
            return calibration_results
        except ImportError:
            return {'error': 'sklearn.calibration not available'}
    
    def get_top_k_errors(self, y_true: List[int], y_pred_proba: np.ndarray, 
                        k: int = 5, worst_n: int = 10) -> Dict[str, Any]:
        """Get top-k prediction errors for analysis"""
        predictions = np.argmax(y_pred_proba, axis=1)
        confidences = np.max(y_pred_proba, axis=1)
        
        # Find incorrect predictions
        incorrect_mask = predictions != np.array(y_true)
        incorrect_indices = np.where(incorrect_mask)[0]
        
        if len(incorrect_indices) == 0:
            return {'message': 'No incorrect predictions found'}
        
        # Get top-k accuracy for incorrect predictions
        top_k_correct = []
        for idx in incorrect_indices:
            true_label = y_true[idx]
            top_k_preds = np.argsort(y_pred_proba[idx])[-k:]
            top_k_correct.append(true_label in top_k_preds)
        
        # Get worst predictions (highest confidence but wrong)
        incorrect_confidences = confidences[incorrect_indices]
        worst_prediction_indices = incorrect_indices[np.argsort(incorrect_confidences)[-worst_n:]]
        
        worst_predictions = []
        for idx in worst_prediction_indices:
            worst_predictions.append({
                'sample_index': int(idx),
                'true_label': int(y_true[idx]),
                'predicted_label': int(predictions[idx]),
                'confidence': float(confidences[idx]),
                'top_k_probabilities': y_pred_proba[idx].tolist()
            })
        
        return {
            'total_incorrect': len(incorrect_indices),
            'top_k_recovery_rate': np.mean(top_k_correct),
            'worst_predictions': worst_predictions
        }
