"""
Model Registry System for Smart Pill Recognition

This module provides a centralized model registry for managing trained models
from different training methods (PyTorch, PySpark, Transformers).

Author: DoAnDLL Team
Date: 2024
"""

import os
import json
import torch
import pickle
import joblib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import yaml
from dataclasses import dataclass, asdict
from enum import Enum
import shutil
import hashlib

class TrainingMethod(Enum):
    """Training method types"""
    PYTORCH = "pytorch"
    PYSPARK = "pyspark"
    TRANSFORMERS = "transformers"

@dataclass
class ModelMetadata:
    """Model metadata structure"""
    model_id: str
    name: str
    training_method: TrainingMethod
    version: str
    created_at: str
    accuracy: float
    loss: float
    training_time: float
    dataset_size: int
    model_path: str
    config_path: str
    metrics_path: str
    description: str
    tags: List[str]
    hyperparameters: Dict[str, Any]
    hardware_info: Dict[str, Any]

class ModelRegistry:
    """
    Central registry for managing trained models from different methods
    """
    
    def __init__(self, registry_path: str = "./models_registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        # Registry database file
        self.db_path = self.registry_path / "registry.json"
        self.models_dir = self.registry_path / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing registry
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Dict]:
        """Load registry database"""
        if self.db_path.exists():
            with open(self.db_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_registry(self):
        """Save registry database"""
        with open(self.db_path, 'w') as f:
            json.dump(self.registry, f, indent=2, default=str)
    
    def _generate_model_id(self, name: str, training_method: TrainingMethod) -> str:
        """Generate unique model ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        method_prefix = training_method.value[:2].upper()
        return f"{method_prefix}_{name}_{timestamp}"
    
    def register_model(self,
                      name: str,
                      training_method: TrainingMethod,
                      model_artifact: Any,
                      config: Dict[str, Any],
                      metrics: Dict[str, Any],
                      description: str = "",
                      tags: List[str] = None,
                      overwrite: bool = False) -> str:
        """
        Register a new trained model
        
        Args:
            name: Model name
            training_method: Training method used
            model_artifact: Model object or path
            config: Training configuration
            metrics: Training metrics
            description: Model description
            tags: Model tags
            overwrite: Whether to overwrite existing model
            
        Returns:
            Model ID
        """
        if tags is None:
            tags = []
        
        # Generate model ID
        model_id = self._generate_model_id(name, training_method)
        
        # Check if model already exists
        if model_id in self.registry and not overwrite:
            raise ValueError(f"Model {model_id} already exists. Use overwrite=True to replace.")
        
        # Create model directory
        model_dir = self.models_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model artifacts
        model_path = self._save_model_artifact(model_artifact, model_dir, training_method)
        config_path = model_dir / "config.yaml"
        metrics_path = model_dir / "metrics.json"
        
        # Save config and metrics
        with open(config_path, 'w') as f:
            yaml.dump(config, f, indent=2)
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            name=name,
            training_method=training_method,
            version="1.0.0",
            created_at=datetime.now().isoformat(),
            accuracy=metrics.get('accuracy', 0.0),
            loss=metrics.get('loss', float('inf')),
            training_time=metrics.get('training_time', 0.0),
            dataset_size=metrics.get('dataset_size', 0),
            model_path=str(model_path.relative_to(self.registry_path)),
            config_path=str(config_path.relative_to(self.registry_path)),
            metrics_path=str(metrics_path.relative_to(self.registry_path)),
            description=description,
            tags=tags,
            hyperparameters=config.get('training', {}),
            hardware_info=config.get('hardware', {})
        )
        
        # Register model
        self.registry[model_id] = asdict(metadata)
        self._save_registry()
        
        print(f"✅ Model registered successfully: {model_id}")
        return model_id
    
    def _save_model_artifact(self, 
                           model_artifact: Any, 
                           model_dir: Path, 
                           training_method: TrainingMethod) -> Path:
        """Save model artifact based on training method"""
        
        if training_method == TrainingMethod.PYTORCH:
            model_path = model_dir / "pytorch_model.pth"
            if hasattr(model_artifact, 'state_dict'):
                # PyTorch model object
                torch.save({
                    'model_state_dict': model_artifact.state_dict(),
                    'model_class': model_artifact.__class__.__name__
                }, model_path)
            else:
                # Assume it's already a state dict or checkpoint
                torch.save(model_artifact, model_path)
                
        elif training_method == TrainingMethod.PYSPARK:
            model_path = model_dir / "spark_model"
            if hasattr(model_artifact, 'save'):
                # PySpark model with save method
                model_artifact.save(str(model_path))
            else:
                # Save as pickle
                model_path = model_dir / "spark_model.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model_artifact, f)
                    
        elif training_method == TrainingMethod.TRANSFORMERS:
            model_path = model_dir / "transformers_model"
            if hasattr(model_artifact, 'save_pretrained'):
                # HuggingFace model
                model_artifact.save_pretrained(model_path)
            else:
                # Save as pickle fallback
                model_path = model_dir / "transformers_model.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model_artifact, f)
        
        return model_path
    
    def load_model(self, model_id: str) -> Dict[str, Any]:
        """
        Load a registered model
        
        Args:
            model_id: Model identifier
            
        Returns:
            Dictionary containing model, config, and metadata
        """
        if model_id not in self.registry:
            raise ValueError(f"Model {model_id} not found in registry")
        
        metadata = self.registry[model_id]
        training_method = TrainingMethod(metadata['training_method'])
        
        # Load config
        config_path = self.registry_path / metadata['config_path']
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Load metrics
        metrics_path = self.registry_path / metadata['metrics_path']
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # Load model based on training method
        model_path = self.registry_path / metadata['model_path']
        model = self._load_model_artifact(model_path, training_method)
        
        return {
            'model': model,
            'config': config,
            'metrics': metrics,
            'metadata': metadata
        }
    
    def _load_model_artifact(self, model_path: Path, training_method: TrainingMethod):
        """Load model artifact based on training method"""
        
        if training_method == TrainingMethod.PYTORCH:
            if model_path.is_file():
                return torch.load(model_path, map_location='cpu')
            else:
                raise FileNotFoundError(f"PyTorch model not found: {model_path}")
                
        elif training_method == TrainingMethod.PYSPARK:
            if model_path.is_dir():
                # Load PySpark model directory
                try:
                    from pyspark.ml import Pipeline
                    from pyspark.ml.classification import MultilayerPerceptronClassifier
                    # Try to load as PySpark model
                    # This is a placeholder - actual implementation depends on model type
                    return str(model_path)  # Return path for now
                except ImportError:
                    pass
            
            # Try pickle fallback
            pickle_path = model_path.parent / "spark_model.pkl"
            if pickle_path.exists():
                with open(pickle_path, 'rb') as f:
                    return pickle.load(f)
            else:
                raise FileNotFoundError(f"Spark model not found: {model_path}")
                
        elif training_method == TrainingMethod.TRANSFORMERS:
            if model_path.is_dir():
                try:
                    from transformers import AutoModel
                    return AutoModel.from_pretrained(model_path)
                except ImportError:
                    pass
            
            # Try pickle fallback
            pickle_path = model_path.parent / "transformers_model.pkl"
            if pickle_path.exists():
                with open(pickle_path, 'rb') as f:
                    return pickle.load(f)
            else:
                raise FileNotFoundError(f"Transformers model not found: {model_path}")
    
    def list_models(self, 
                   training_method: Optional[TrainingMethod] = None,
                   tags: Optional[List[str]] = None,
                   sort_by: str = "created_at",
                   reverse: bool = True) -> List[Dict[str, Any]]:
        """
        List registered models with filtering
        
        Args:
            training_method: Filter by training method
            tags: Filter by tags
            sort_by: Sort field
            reverse: Sort order
            
        Returns:
            List of model metadata
        """
        models = list(self.registry.values())
        
        # Filter by training method
        if training_method:
            models = [m for m in models if m['training_method'] == training_method.value]
        
        # Filter by tags
        if tags:
            models = [m for m in models if any(tag in m['tags'] for tag in tags)]
        
        # Sort models
        if sort_by in ['accuracy', 'loss', 'training_time']:
            models.sort(key=lambda x: x.get(sort_by, 0), reverse=reverse)
        else:
            models.sort(key=lambda x: x.get(sort_by, ''), reverse=reverse)
        
        return models
    
    def get_best_model(self, 
                      metric: str = "accuracy",
                      training_method: Optional[TrainingMethod] = None) -> Optional[Dict[str, Any]]:
        """
        Get best model based on a metric
        
        Args:
            metric: Metric to optimize (accuracy, loss, etc.)
            training_method: Filter by training method
            
        Returns:
            Best model metadata or None
        """
        models = self.list_models(training_method=training_method)
        
        if not models:
            return None
        
        if metric == "loss":
            # Lower is better for loss
            best_model = min(models, key=lambda x: x.get(metric, float('inf')))
        else:
            # Higher is better for accuracy, etc.
            best_model = max(models, key=lambda x: x.get(metric, 0))
        
        return best_model
    
    def delete_model(self, model_id: str, confirm: bool = False):
        """
        Delete a registered model
        
        Args:
            model_id: Model identifier
            confirm: Confirmation flag
        """
        if model_id not in self.registry:
            raise ValueError(f"Model {model_id} not found in registry")
        
        if not confirm:
            print(f"⚠️  This will permanently delete model {model_id}")
            print("Use confirm=True to proceed")
            return
        
        # Remove model directory
        model_dir = self.models_dir / model_id
        if model_dir.exists():
            shutil.rmtree(model_dir)
        
        # Remove from registry
        del self.registry[model_id]
        self._save_registry()
        
        print(f"✅ Model {model_id} deleted successfully")
    
    def export_model(self, model_id: str, export_path: str):
        """
        Export model for deployment
        
        Args:
            model_id: Model identifier
            export_path: Export destination
        """
        if model_id not in self.registry:
            raise ValueError(f"Model {model_id} not found in registry")
        
        export_dir = Path(export_path)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy model directory
        model_dir = self.models_dir / model_id
        shutil.copytree(model_dir, export_dir / model_id, dirs_exist_ok=True)
        
        # Create deployment info
        deployment_info = {
            'model_id': model_id,
            'metadata': self.registry[model_id],
            'exported_at': datetime.now().isoformat()
        }
        
        with open(export_dir / f"{model_id}_deployment.json", 'w') as f:
            json.dump(deployment_info, f, indent=2, default=str)
        
        print(f"✅ Model {model_id} exported to {export_path}")
    
    def get_comparison_data(self) -> List[Dict[str, Any]]:
        """
        Get data for comparing different training methods
        
        Returns:
            List of models with comparison metrics
        """
        models = list(self.registry.values())
        
        comparison_data = []
        for model in models:
            comparison_data.append({
                'model_id': model['model_id'],
                'name': model['name'],
                'training_method': model['training_method'],
                'accuracy': model['accuracy'],
                'loss': model['loss'],
                'training_time': model['training_time'],
                'dataset_size': model['dataset_size'],
                'created_at': model['created_at']
            })
        
        return comparison_data
    
    def generate_comparison_report(self, output_path: str = None) -> Dict[str, Any]:
        """
        Generate comprehensive comparison report
        
        Args:
            output_path: Optional path to save report
            
        Returns:
            Comparison report
        """
        comparison_data = self.get_comparison_data()
        
        if not comparison_data:
            return {"error": "No models found for comparison"}
        
        # Group by training method
        methods = {}
        for model in comparison_data:
            method = model['training_method']
            if method not in methods:
                methods[method] = []
            methods[method].append(model)
        
        # Calculate statistics for each method
        method_stats = {}
        for method, models in methods.items():
            accuracies = [m['accuracy'] for m in models]
            losses = [m['loss'] for m in models if m['loss'] != float('inf')]
            training_times = [m['training_time'] for m in models]
            
            method_stats[method] = {
                'count': len(models),
                'avg_accuracy': sum(accuracies) / len(accuracies) if accuracies else 0,
                'max_accuracy': max(accuracies) if accuracies else 0,
                'min_accuracy': min(accuracies) if accuracies else 0,
                'avg_loss': sum(losses) / len(losses) if losses else float('inf'),
                'avg_training_time': sum(training_times) / len(training_times) if training_times else 0,
                'models': models
            }
        
        # Generate report
        report = {
            'generated_at': datetime.now().isoformat(),
            'total_models': len(comparison_data),
            'training_methods': list(methods.keys()),
            'method_statistics': method_stats,
            'best_models': {
                'overall_accuracy': max(comparison_data, key=lambda x: x['accuracy'], default={}),
                'overall_speed': min(comparison_data, key=lambda x: x['training_time'], default={}),
            }
        }
        
        # Save report if path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"✅ Comparison report saved to {output_path}")
        
        return report


# Convenience functions
def get_registry(registry_path: str = "./models_registry") -> ModelRegistry:
    """Get model registry instance"""
    return ModelRegistry(registry_path)

def register_pytorch_model(model, config: Dict, metrics: Dict, name: str, **kwargs) -> str:
    """Quick registration for PyTorch models"""
    registry = get_registry()
    return registry.register_model(
        name=name,
        training_method=TrainingMethod.PYTORCH,
        model_artifact=model,
        config=config,
        metrics=metrics,
        **kwargs
    )

def register_spark_model(model, config: Dict, metrics: Dict, name: str, **kwargs) -> str:
    """Quick registration for Spark models"""
    registry = get_registry()
    return registry.register_model(
        name=name,
        training_method=TrainingMethod.PYSPARK,
        model_artifact=model,
        config=config,
        metrics=metrics,
        **kwargs
    )

def register_transformers_model(model, config: Dict, metrics: Dict, name: str, **kwargs) -> str:
    """Quick registration for Transformers models"""
    registry = get_registry()
    return registry.register_model(
        name=name,
        training_method=TrainingMethod.TRANSFORMERS,
        model_artifact=model,
        config=config,
        metrics=metrics,
        **kwargs
    )