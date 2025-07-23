"""
HuggingFace Transformers Training Module for Smart Pill Recognition

This module implements state-of-the-art multimodal training using HuggingFace
Transformers library with advanced features like custom trainers, callbacks,
and model optimization.

Author: DoAnDLL Team
Date: 2024
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass

# HuggingFace imports
try:
    from transformers import (
        AutoModel, AutoTokenizer, AutoProcessor, AutoConfig,
        Trainer, TrainingArguments, 
        EvalPrediction, TrainerCallback,
        get_linear_schedule_with_warmup,
        set_seed
    )
    from transformers.modeling_outputs import SequenceClassifierOutput
    from transformers.models.auto.modeling_auto import MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING
    
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    print(f"Transformers not available: {e}")
    TRANSFORMERS_AVAILABLE = False

# Additional ML libraries
try:
    from datasets import Dataset, DatasetDict
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    import evaluate
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

# Vision libraries
try:
    from PIL import Image
    import torchvision.transforms as transforms
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False

from ..models.model_registry import ModelRegistry, TrainingMethod
from ..utils.metrics import MetricsCalculator


class MultimodalPillConfig(AutoConfig):
    """Configuration for multimodal pill recognition model"""
    
    model_type = "multimodal_pill"
    
    def __init__(self,
                 vision_model_name: str = "google/vit-base-patch16-224",
                 text_model_name: str = "bert-base-uncased", 
                 num_classes: int = 1000,
                 hidden_dropout_prob: float = 0.1,
                 fusion_type: str = "cross_attention",
                 image_size: int = 224,
                 max_text_length: int = 128,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.vision_model_name = vision_model_name
        self.text_model_name = text_model_name
        self.num_classes = num_classes
        self.hidden_dropout_prob = hidden_dropout_prob
        self.fusion_type = fusion_type
        self.image_size = image_size
        self.max_text_length = max_text_length


class MultimodalPillModel(nn.Module):
    """
    HuggingFace-style multimodal model for pill recognition
    """
    
    def __init__(self, config: MultimodalPillConfig):
        super().__init__()
        self.config = config
        self.num_labels = config.num_classes
        
        # Load vision model
        self.vision_model = AutoModel.from_pretrained(
            config.vision_model_name,
            add_pooling_layer=False
        )
        
        # Load text model  
        self.text_model = AutoModel.from_pretrained(
            config.text_model_name,
            add_pooling_layer=False
        )
        
        # Get hidden sizes
        self.vision_hidden_size = self.vision_model.config.hidden_size
        self.text_hidden_size = self.text_model.config.hidden_size
        
        # Cross-modal attention
        if config.fusion_type == "cross_attention":
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=self.vision_hidden_size,
                num_heads=8,
                dropout=config.hidden_dropout_prob,
                batch_first=True
            )
            
            # Projection layer if needed
            if self.text_hidden_size != self.vision_hidden_size:
                self.text_projection = nn.Linear(self.text_hidden_size, self.vision_hidden_size)
            else:
                self.text_projection = nn.Identity()
        
        # Fusion layer
        if config.fusion_type == "concat":
            fusion_dim = self.vision_hidden_size + self.text_hidden_size
        else:
            fusion_dim = self.vision_hidden_size
            
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(512, 256),
            nn.GELU(), 
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(256, self.num_labels)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self,
                pixel_values: torch.Tensor = None,
                input_ids: torch.Tensor = None,
                attention_mask: torch.Tensor = None,
                labels: torch.Tensor = None,
                **kwargs) -> Union[Tuple, SequenceClassifierOutput]:
        """
        Forward pass for multimodal model
        
        Args:
            pixel_values: Image tensor [batch_size, channels, height, width]
            input_ids: Text input IDs [batch_size, seq_len]
            attention_mask: Text attention mask [batch_size, seq_len]
            labels: Labels for training [batch_size]
            
        Returns:
            Model outputs
        """
        # Vision encoding
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        vision_features = vision_outputs.last_hidden_state  # [batch_size, num_patches, hidden_size]
        
        # Text encoding
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_features = text_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Feature fusion
        if self.config.fusion_type == "cross_attention":
            # Project text features if needed
            text_features_proj = self.text_projection(text_features)
            
            # Cross-modal attention (vision queries, text keys/values)
            attended_vision, _ = self.cross_attention(
                query=vision_features,
                key=text_features_proj,
                value=text_features_proj,
                key_padding_mask=~attention_mask.bool() if attention_mask is not None else None
            )
            
            # Global average pooling
            fused_features = attended_vision.mean(dim=1)  # [batch_size, hidden_size]
            
        elif self.config.fusion_type == "concat":
            # Simple concatenation with pooling
            vision_pooled = vision_features.mean(dim=1)  # [batch_size, vision_hidden_size]
            
            # Text pooling with attention mask
            if attention_mask is not None:
                text_mask = attention_mask.unsqueeze(-1).float()
                text_pooled = (text_features * text_mask).sum(dim=1) / text_mask.sum(dim=1)
            else:
                text_pooled = text_features.mean(dim=1)
            
            fused_features = torch.cat([vision_pooled, text_pooled], dim=-1)
        
        else:
            # Simple average fusion
            vision_pooled = vision_features.mean(dim=1)
            text_pooled = text_features.mean(dim=1)
            fused_features = (vision_pooled + text_pooled) / 2
        
        # Classification
        logits = self.classifier(fused_features)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None
        )


class MultimodalDataset:
    """Dataset class for multimodal pill recognition"""
    
    def __init__(self,
                 data: List[Dict],
                 tokenizer,
                 image_processor,
                 max_length: int = 128,
                 image_size: int = 224):
        self.data = data
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.image_size = image_size
        
        # Image transforms
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Process image
        if 'image_path' in item:
            image = Image.open(item['image_path']).convert('RGB')
        elif 'image' in item:
            image = item['image'].convert('RGB')
        else:
            # Create dummy image if none provided
            image = Image.new('RGB', (self.image_size, self.image_size), color=(128, 128, 128))
        
        pixel_values = self.image_transform(image)
        
        # Process text
        text = item.get('text', item.get('text_imprint', ''))
        if not text:
            text = ""  # Empty text if none provided
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Get label
        label = item.get('label', 0)
        
        return {
            'pixel_values': pixel_values.squeeze(0),
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class MultimodalTrainingArguments(TrainingArguments):
    """Extended training arguments for multimodal training"""
    
    def __init__(self, 
                 image_size: int = 224,
                 max_text_length: int = 128,
                 fusion_type: str = "cross_attention",
                 **kwargs):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.max_text_length = max_text_length
        self.fusion_type = fusion_type


class MultimodalTrainer(Trainer):
    """Custom trainer for multimodal pill recognition"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_calculator = MetricsCalculator(
            num_classes=self.model.num_labels
        )
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute loss for training"""
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Compute custom loss if needed
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.model.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Enhanced evaluation with custom metrics"""
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        
        # Evaluation loop
        model = self.model
        model.eval()
        
        all_predictions = []
        all_labels = []
        total_loss = 0.0
        
        for batch in eval_dataloader:
            batch = self._prepare_inputs(batch)
            
            with torch.no_grad():
                outputs = model(**batch)
                loss = outputs.loss
                logits = outputs.logits
                
                total_loss += loss.item()
                
                predictions = torch.argmax(logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_metrics(all_labels, all_predictions)
        metrics['loss'] = total_loss / len(eval_dataloader)
        
        # Add prefix to metrics
        prefixed_metrics = {f"{metric_key_prefix}_{k}": v for k, v in metrics.items()}
        
        return prefixed_metrics


class PillRecognitionCallback(TrainerCallback):
    """Custom callback for pill recognition training"""
    
    def __init__(self, model_registry: ModelRegistry, model_name: str):
        self.model_registry = model_registry
        self.model_name = model_name
        self.best_accuracy = 0.0
    
    def on_evaluate(self, args, state, control, model, logs=None, **kwargs):
        """Save best model during training"""
        if logs and 'eval_accuracy' in logs:
            accuracy = logs['eval_accuracy']
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                print(f"🎯 New best accuracy: {accuracy:.4f}")


class HuggingFaceMultimodalTrainer:
    """
    Main trainer class using HuggingFace ecosystem
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_registry = ModelRegistry()
        
        # Initialize tokenizer and processor
        model_config = config.get('model', {})
        self.text_model_name = model_config.get('text_encoder', {}).get('model_name', 'bert-base-uncased')
        self.vision_model_name = model_config.get('visual_encoder', {}).get('model_name', 'google/vit-base-patch16-224')
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.text_model_name)
        
        try:
            self.image_processor = AutoProcessor.from_pretrained(self.vision_model_name)
        except:
            self.image_processor = None
        
        print("✅ HuggingFace Multimodal Trainer initialized")
    
    def prepare_dataset(self, 
                       data: List[Dict],
                       train_split: float = 0.8,
                       val_split: float = 0.1) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Prepare datasets for training
        
        Args:
            data: List of data samples
            train_split: Training split ratio
            val_split: Validation split ratio
            
        Returns:
            Train, validation, and test datasets
        """
        # Split data
        n = len(data)
        train_size = int(n * train_split)
        val_size = int(n * val_split)
        
        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]
        
        # Create datasets
        image_size = self.config.get('data', {}).get('image_size', 224)
        max_length = self.config.get('model', {}).get('text_encoder', {}).get('max_length', 128)
        
        train_dataset = MultimodalDataset(
            train_data, self.tokenizer, self.image_processor,
            max_length=max_length, image_size=image_size
        )
        
        val_dataset = MultimodalDataset(
            val_data, self.tokenizer, self.image_processor,
            max_length=max_length, image_size=image_size
        )
        
        test_dataset = MultimodalDataset(
            test_data, self.tokenizer, self.image_processor,
            max_length=max_length, image_size=image_size
        )
        
        print(f"📊 Dataset splits - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset
    
    def create_model(self) -> MultimodalPillModel:
        """Create multimodal model"""
        model_config_dict = self.config.get('model', {})
        
        config = MultimodalPillConfig(
            vision_model_name=self.vision_model_name,
            text_model_name=self.text_model_name,
            num_classes=model_config_dict.get('classifier', {}).get('num_classes', 1000),
            hidden_dropout_prob=model_config_dict.get('dropout', 0.1),
            fusion_type=model_config_dict.get('fusion', {}).get('type', 'cross_attention'),
            image_size=self.config.get('data', {}).get('image_size', 224),
            max_text_length=model_config_dict.get('text_encoder', {}).get('max_length', 128)
        )
        
        model = MultimodalPillModel(config)
        
        print(f"✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        return model
    
    def train_model(self, 
                   train_dataset: Dataset,
                   val_dataset: Dataset,
                   model_name: str = "hf_multimodal_pill") -> Dict[str, Any]:
        """
        Train model using HuggingFace Trainer
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            model_name: Name for saved model
            
        Returns:
            Training results
        """
        start_time = time.time()
        
        try:
            # Create model
            model = self.create_model()
            
            # Training arguments
            training_config = self.config.get('training', {})
            
            training_args = MultimodalTrainingArguments(
                output_dir=f"./results/{model_name}",
                num_train_epochs=training_config.get('num_epochs', 10),
                per_device_train_batch_size=training_config.get('batch_size', 16),
                per_device_eval_batch_size=training_config.get('batch_size', 16),
                warmup_steps=training_config.get('warmup_steps', 500),
                weight_decay=training_config.get('weight_decay', 0.01),
                learning_rate=training_config.get('learning_rate', 5e-5),
                logging_dir=f'./logs/{model_name}',
                logging_steps=100,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                save_total_limit=3,
                load_best_model_at_end=True,
                metric_for_best_model="eval_accuracy",
                greater_is_better=True,
                fp16=training_config.get('mixed_precision', False),
                dataloader_num_workers=4,
                remove_unused_columns=False,  # Keep all columns for multimodal data
                image_size=self.config.get('data', {}).get('image_size', 224),
                max_text_length=training_config.get('max_length', 128),
                fusion_type=self.config.get('model', {}).get('fusion', {}).get('type', 'cross_attention')
            )
            
            # Custom callback
            callback = PillRecognitionCallback(self.model_registry, model_name)
            
            # Create trainer
            trainer = MultimodalTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                callbacks=[callback]
            )
            
            # Train model
            print("🏋️ Starting HuggingFace training...")
            train_result = trainer.train()
            
            # Final evaluation
            print("📊 Performing final evaluation...")
            eval_results = trainer.evaluate()
            
            # Save model
            trainer.save_model()
            self.tokenizer.save_pretrained(training_args.output_dir)
            
            # Calculate training time
            training_time = time.time() - start_time
            
            # Prepare results
            results = {
                'model': trainer.model,
                'trainer': trainer,
                'train_results': train_result,
                'eval_results': eval_results,
                'training_time': training_time,
                'dataset_size': len(train_dataset),
                'model_path': training_args.output_dir
            }
            
            # Register model
            metrics = {
                'accuracy': eval_results.get('eval_accuracy', 0.0),
                'loss': eval_results.get('eval_loss', float('inf')),
                'training_time': training_time,
                'dataset_size': len(train_dataset)
            }
            
            model_id = self.model_registry.register_model(
                name=model_name,
                training_method=TrainingMethod.TRANSFORMERS,
                model_artifact=trainer.model,
                config=self.config,
                metrics=metrics,
                description=f"HuggingFace multimodal model with {self.text_model_name} and {self.vision_model_name}",
                tags=["huggingface", "transformers", "multimodal", "state-of-art"]
            )
            
            results['model_id'] = model_id
            
            print(f"✅ HuggingFace training completed!")
            print(f"📊 Model ID: {model_id}")
            print(f"🎯 Final Accuracy: {eval_results.get('eval_accuracy', 0.0):.4f}")
            print(f"⏱️  Training time: {training_time:.2f}s")
            
            return results
            
        except Exception as e:
            print(f"❌ HuggingFace training failed: {e}")
            import traceback
            traceback.print_exc()
            return {}


def create_hf_trainer(config_path: str = None) -> HuggingFaceMultimodalTrainer:
    """
    Create HuggingFace trainer with configuration
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configured HuggingFaceMultimodalTrainer
    """
    if config_path:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default configuration
        config = {
            "model": {
                "visual_encoder": {
                    "model_name": "google/vit-base-patch16-224"
                },
                "text_encoder": {
                    "model_name": "bert-base-uncased",
                    "max_length": 128
                },
                "fusion": {
                    "type": "cross_attention"
                },
                "classifier": {
                    "num_classes": 1000
                },
                "dropout": 0.1
            },
            "training": {
                "num_epochs": 10,
                "batch_size": 16,
                "learning_rate": 5e-5,
                "weight_decay": 0.01,
                "warmup_steps": 500,
                "mixed_precision": True
            },
            "data": {
                "image_size": 224
            }
        }
    
    return HuggingFaceMultimodalTrainer(config)


def train_hf_model(data: List[Dict],
                  config_path: str = None,
                  model_name: str = "hf_pill_model") -> Dict[str, Any]:
    """
    Complete HuggingFace training pipeline
    
    Args:
        data: Training data
        config_path: Configuration file path
        model_name: Name for saved model
        
    Returns:
        Training results
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("Transformers not available. Please install transformers.")
    
    try:
        # Create trainer
        trainer = create_hf_trainer(config_path)
        
        # Prepare datasets
        train_dataset, val_dataset, test_dataset = trainer.prepare_dataset(data)
        
        # Train model
        results = trainer.train_model(train_dataset, val_dataset, model_name)
        
        return results
        
    except Exception as e:
        print(f"❌ HuggingFace training pipeline failed: {e}")
        return {}


if __name__ == "__main__":
    # Example usage
    print("🧪 Testing HuggingFace trainer...")
    
    # Sample data
    sample_data = [
        {
            'image_path': '/tmp/dummy_image.jpg',
            'text': 'ADVIL 200',
            'label': 0
        }
    ]
    
    # Create dummy image for testing
    if VISION_AVAILABLE:
        dummy_image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        dummy_image.save('/tmp/dummy_image.jpg')
    
    config = {
        "model": {
            "classifier": {"num_classes": 10}
        },
        "training": {
            "num_epochs": 1,
            "batch_size": 2
        }
    }
    
    if TRANSFORMERS_AVAILABLE:
        trainer = HuggingFaceMultimodalTrainer(config)
        print("✅ HuggingFace trainer created successfully")