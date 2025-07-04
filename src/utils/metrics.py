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
        for i in range(len(precision_per_class)):
            per_class_metrics[i] = {
                'precision': precision_per_class[i],
                'recall': recall_per_class[i],
                'f1': f1_per_class[i]
            }
        
        return per_class_metrics


def compute_multimodal_metrics(visual_features: torch.Tensor, text_features: torch.Tensor,
                              fused_features: torch.Tensor) -> Dict[str, float]:
    """Compute metrics specific to multimodal learning"""
    
    metrics = {}
    
    # Feature similarity between modalities
    visual_norm = torch.nn.functional.normalize(visual_features, dim=1)
    text_norm = torch.nn.functional.normalize(text_features, dim=1)
    
    # Cosine similarity between visual and text features
    cosine_sim = torch.mean(torch.sum(visual_norm * text_norm, dim=1))
    metrics['visual_text_similarity'] = cosine_sim.item()
    
    # Feature magnitude
    metrics['visual_feature_magnitude'] = torch.mean(torch.norm(visual_features, dim=1)).item()
    metrics['text_feature_magnitude'] = torch.mean(torch.norm(text_features, dim=1)).item()
    metrics['fused_feature_magnitude'] = torch.mean(torch.norm(fused_features, dim=1)).item()
    
    return metrics
