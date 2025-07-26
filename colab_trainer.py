#!/usr/bin/env python3
"""
Colab-Compatible Training Module for Smart Pill Recognition System

This module provides training functionality specifically optimized for Google Colab,
with proper dependency handling and fallback mechanisms.

Author: DoAnDLL Team  
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import warnings

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Suppress warnings
warnings.filterwarnings("ignore")

def check_colab_environment():
    """Check if running in Google Colab"""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def setup_colab_environment():
    """Setup Google Colab environment with necessary installations"""
    if not check_colab_environment():
        return False
    
    print("🔧 Setting up Google Colab environment...")
    
    # Install required packages
    install_commands = [
        "pip install -q transformers datasets timm",
        "pip install -q opencv-python-headless Pillow",
        "pip install -q scikit-learn matplotlib seaborn",
        "pip install -q tqdm rich accelerate"
    ]
    
    for cmd in install_commands:
        try:
            os.system(cmd)
        except Exception as e:
            print(f"Warning: Failed to install packages: {e}")
    
    print("✅ Colab environment setup completed!")
    return True

class ColabMultimodalPillTransformer(nn.Module):
    """
    Multimodal Transformer optimized for Google Colab
    Combines Vision Transformer and BERT for pill recognition
    """
    
    def __init__(self, num_classes=1000, hidden_dim=768, dropout=0.1):
        super().__init__()
        
        # Import transformers with error handling
        try:
            from transformers import ViTModel, BertModel
            
            # Vision Encoder (ViT)
            self.vision_encoder = ViTModel.from_pretrained(
                'google/vit-base-patch16-224',
                add_pooling_layer=False
            )
            
            # Text Encoder (BERT)
            self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
            
        except ImportError as e:
            print(f"Warning: Transformers not available: {e}")
            # Fallback to simple CNN + RNN
            self.vision_encoder = self._create_simple_vision_encoder()
            self.text_encoder = self._create_simple_text_encoder()
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Fusion layers
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
    
    def _create_simple_vision_encoder(self):
        """Fallback vision encoder if ViT is not available"""
        return nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 768)
        )
    
    def _create_simple_text_encoder(self):
        """Fallback text encoder if BERT is not available"""
        return nn.Sequential(
            nn.Embedding(30522, 768),  # BERT vocab size
            nn.LSTM(768, 768, batch_first=True),
        )
    
    def forward(self, images, input_ids, attention_mask):
        # Handle different encoder types
        if hasattr(self.vision_encoder, 'config'):
            # Transformer-based encoder
            vision_outputs = self.vision_encoder(images)
            image_features = vision_outputs.last_hidden_state
        else:
            # Simple CNN encoder
            image_features = self.vision_encoder(images)
            image_features = image_features.unsqueeze(1)  # Add sequence dimension
        
        if hasattr(self.text_encoder, 'config'):
            # BERT encoder
            text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            text_features = text_outputs.last_hidden_state
        else:
            # Simple LSTM encoder
            embeddings = self.text_encoder[0](input_ids)
            text_features, _ = self.text_encoder[1](embeddings)
        
        # Cross-modal attention
        try:
            attended_image, _ = self.cross_attention(
                query=image_features,
                key=text_features,
                value=text_features,
                key_padding_mask=~attention_mask.bool() if attention_mask is not None else None
            )
        except:
            # Fallback to simple concatenation
            attended_image = image_features
        
        # Global pooling
        image_pooled = attended_image.mean(dim=1)
        text_pooled = text_features.mean(dim=1)
        
        # Fusion
        fused_features = torch.cat([image_pooled, text_pooled], dim=-1)
        fused_features = self.fusion_layer(fused_features)
        
        # Classification
        logits = self.classifier(fused_features)
        
        return logits

class ColabTrainer:
    """Training class optimized for Google Colab"""
    
    def __init__(self, model, device='auto', mixed_precision=True):
        self.model = model
        
        # Auto-detect device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device('cpu')
                print("⚠️ Using CPU (GPU not available)")
        else:
            self.device = torch.device(device)
        
        self.model = self.model.to(self.device)
        self.mixed_precision = mixed_precision and torch.cuda.is_available()
        
        if self.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
            print("✅ Mixed precision training enabled")
    
    def train(self, train_dataloader, val_dataloader=None, epochs=10, learning_rate=2e-5, 
              save_path='/content/checkpoints', patience=5):
        """
        Train the model with comprehensive logging and checkpointing
        """
        
        # Setup
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()
        
        # Create save directory
        os.makedirs(save_path, exist_ok=True)
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        best_val_acc = 0.0
        patience_counter = 0
        start_time = time.time()
        
        print(f"🚀 Starting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.mixed_precision}")
        print(f"Learning Rate: {learning_rate}")
        
        try:
            from tqdm.auto import tqdm
        except ImportError:
            # Fallback progress tracking
            class tqdm:
                def __init__(self, iterable, desc="", total=None):
                    self.iterable = iterable
                    self.desc = desc
                    self.total = total or len(iterable)
                    self.current = 0
                
                def __iter__(self):
                    for item in self.iterable:
                        yield item
                        self.current += 1
                        if self.current % 10 == 0:
                            print(f"{self.desc}: {self.current}/{self.total}")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            
            for batch in train_bar:
                # Move to device
                images = batch['pixel_values'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass with mixed precision
                if self.mixed_precision:
                    with torch.cuda.amp.autocast():
                        logits = self.model(images, input_ids, attention_mask)
                        loss = criterion(logits, labels)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    logits = self.model(images, input_ids, attention_mask)
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()
                
                # Calculate accuracy
                predictions = torch.argmax(logits, dim=-1)
                correct = (predictions == labels).sum().item()
                
                # Update metrics
                train_loss += loss.item()
                train_correct += correct
                train_total += len(labels)
                
                # Update progress bar
                current_acc = train_correct / train_total
                train_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{current_acc:.4f}'
                })
            
            # Calculate epoch metrics
            epoch_train_loss = train_loss / len(train_dataloader)
            epoch_train_acc = train_correct / train_total
            
            # Validation phase
            if val_dataloader is not None:
                val_loss, val_acc = self._validate(val_dataloader, criterion)
            else:
                val_loss, val_acc = epoch_train_loss, epoch_train_acc
            
            # Update scheduler
            scheduler.step()
            
            # Save history
            history['train_loss'].append(epoch_train_loss)
            history['train_acc'].append(epoch_train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['learning_rate'].append(scheduler.get_last_lr()[0])
            
            # Print epoch results
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.2e}")
            
            # Early stopping and checkpointing
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                # Save best model
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_acc': best_val_acc,
                    'history': history,
                    'model_config': {
                        'num_classes': self.model.num_classes,
                        'hidden_dim': self.model.hidden_dim
                    }
                }
                
                checkpoint_path = os.path.join(save_path, 'best_model.pth')
                torch.save(checkpoint, checkpoint_path)
                print(f"  ✅ New best model saved: {val_acc:.4f}")
                
            else:
                patience_counter += 1
                print(f"  ⏳ No improvement ({patience_counter}/{patience})")
                
                if patience_counter >= patience:
                    print(f"  🛑 Early stopping triggered!")
                    break
            
            print("-" * 50)
        
        # Training completed
        total_time = time.time() - start_time
        print(f"🎉 Training completed!")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        
        # Save final results
        results = {
            'best_val_acc': best_val_acc,
            'total_time': total_time,
            'epochs_trained': epoch + 1,
            'history': history,
            'device': str(self.device),
            'mixed_precision': self.mixed_precision
        }
        
        results_path = os.path.join(save_path, 'training_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def _validate(self, val_dataloader, criterion):
        """Validation step"""
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                images = batch['pixel_values'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                if self.mixed_precision:
                    with torch.cuda.amp.autocast():
                        logits = self.model(images, input_ids, attention_mask)
                        loss = criterion(logits, labels)
                else:
                    logits = self.model(images, input_ids, attention_mask)
                    loss = criterion(logits, labels)
                
                predictions = torch.argmax(logits, dim=-1)
                correct = (predictions == labels).sum().item()
                
                val_loss += loss.item()
                val_correct += correct
                val_total += len(labels)
        
        avg_val_loss = val_loss / len(val_dataloader)
        avg_val_acc = val_correct / val_total
        
        return avg_val_loss, avg_val_acc

def create_colab_trainer(num_classes=1000, device='auto'):
    """Factory function to create a Colab-optimized trainer"""
    
    # Setup environment if in Colab
    if check_colab_environment():
        setup_colab_environment()
    
    # Create model
    model = ColabMultimodalPillTransformer(num_classes=num_classes)
    
    # Create trainer
    trainer = ColabTrainer(model, device=device)
    
    return trainer, model

def main():
    """Main function for standalone training"""
    print("🔧 Initializing Colab training environment...")
    
    # Check environment
    if check_colab_environment():
        print("✅ Running in Google Colab")
    else:
        print("⚠️ Not running in Colab, some features may not work")
    
    # Create trainer
    trainer, model = create_colab_trainer()
    
    print("✅ Trainer created successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return trainer, model

if __name__ == "__main__":
    trainer, model = main()