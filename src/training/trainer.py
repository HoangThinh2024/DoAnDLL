import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import wandb
from loguru import logger
from typing import Dict, Any, Optional, Tuple
import yaml
from pathlib import Path

from ..models.multimodal_transformer import MultimodalPillTransformer
from ..data.data_processing import create_dataloaders, SparkDataProcessor
from ..utils.metrics import MetricsCalculator
from ..utils.utils import set_seed, save_checkpoint, load_checkpoint


class MultimodalTrainer:
    """Trainer for multimodal pill recognition model"""
    
    def __init__(self, config: Dict[str, Any], resume_from: Optional[str] = None):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Set random seed
        set_seed(42)
        
        # Initialize model
        self.model = MultimodalPillTransformer(config["model"]).to(self.device)
        logger.info(f"Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")
        
        # Initialize optimizer and scheduler
        self._setup_optimizer_and_scheduler()
        
        # Initialize metrics calculator
        self.metrics_calculator = MetricsCalculator(
            num_classes=config["model"]["classifier"]["num_classes"]
        )
        
        # Mixed precision training
        self.scaler = GradScaler() if self.device.type == "cuda" else None
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        # Initialize wandb if configured
        if config["logging"]["wandb"]["project"]:
            wandb.init(
                project=config["logging"]["wandb"]["project"],
                entity=config["logging"]["wandb"]["entity"],
                config=config,
                name=f"multimodal_pill_transformer_{config['model']['name']}"
            )
            wandb.watch(self.model)
        
        # Resume from checkpoint if specified
        if resume_from:
            self.load_checkpoint(resume_from)
    
    def _setup_optimizer_and_scheduler(self):
        """Setup optimizer and learning rate scheduler"""
        
        # Optimizer
        if self.config["training"]["optimizer"].lower() == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config["training"]["learning_rate"],
                weight_decay=self.config["training"]["weight_decay"]
            )
        elif self.config["training"]["optimizer"].lower() == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config["training"]["learning_rate"],
                weight_decay=self.config["training"]["weight_decay"]
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config['training']['optimizer']}")
        
        # Scheduler
        if self.config["training"]["scheduler"] == "cosine_annealing":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config["training"]["num_epochs"]
            )
        elif self.config["training"]["scheduler"] == "onecycle":
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config["training"]["learning_rate"],
                total_steps=self.config["training"]["num_epochs"] * 100  # Approximate
            )
        else:
            self.scheduler = None
    
    def train_epoch(self, dataloader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        
        # Get tokenizer
        tokenizer = self.model.get_text_tokenizer()
        
        pbar = tqdm(dataloader, desc=f"Training Epoch {self.current_epoch}")
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            images = batch['images'].to(self.device)
            texts = batch['texts']
            labels = batch['labels'].to(self.device)
            
            # Tokenize texts
            text_inputs = tokenizer(
                texts,
                max_length=self.config["model"]["text_encoder"]["max_length"],
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.scaler is not None:
                with autocast():
                    outputs = self.model(images, text_inputs)
                    loss = nn.CrossEntropyLoss()(outputs["logits"], labels)
                
                # Backward pass with scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config["training"]["gradient_clip_norm"]
                )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images, text_inputs)
                loss = nn.CrossEntropyLoss()(outputs["logits"], labels)
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config["training"]["gradient_clip_norm"]
                )
                
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss / total_samples:.4f}'
            })
            
            # Log to wandb
            if wandb.run:
                wandb.log({
                    "train_loss_step": loss.item(),
                    "learning_rate": self.optimizer.param_groups[0]['lr']
                })
        
        avg_loss = total_loss / total_samples
        return {"loss": avg_loss}
    
    def validate(self, dataloader) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        # Get tokenizer
        tokenizer = self.model.get_text_tokenizer()
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Validation")
            for batch in pbar:
                # Move data to device
                images = batch['images'].to(self.device)
                texts = batch['texts']
                labels = batch['labels'].to(self.device)
                
                # Tokenize texts
                text_inputs = tokenizer(
                    texts,
                    max_length=self.config["model"]["text_encoder"]["max_length"],
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
                
                # Forward pass
                if self.scaler is not None:
                    with autocast():
                        outputs = self.model(images, text_inputs)
                        loss = nn.CrossEntropyLoss()(outputs["logits"], labels)
                else:
                    outputs = self.model(images, text_inputs)
                    loss = nn.CrossEntropyLoss()(outputs["logits"], labels)
                
                # Get predictions
                predictions = torch.argmax(outputs["logits"], dim=1)
                
                # Accumulate results
                total_loss += loss.item() * images.size(0)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(dataloader.dataset)
        metrics = self.metrics_calculator.calculate_metrics(
            all_labels, all_predictions
        )
        
        metrics["loss"] = avg_loss
        return metrics
    
    def train(self, train_dataloader, val_dataloader):
        """Main training loop"""
        logger.info("Starting training...")
        
        for epoch in range(self.current_epoch, self.config["training"]["num_epochs"]):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch(train_dataloader)
            
            # Validation
            val_metrics = self.validate(val_dataloader)
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, CosineAnnealingLR):
                    self.scheduler.step()
                else:
                    self.scheduler.step()
            
            # Logging
            logger.info(
                f"Epoch {epoch}: "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}"
            )
            
            # Log to wandb
            if wandb.run:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_metrics['loss'],
                    "val_loss": val_metrics['loss'],
                    "val_accuracy": val_metrics['accuracy'],
                    "val_f1": val_metrics['f1_macro'],
                    "val_precision": val_metrics['precision_macro'],
                    "val_recall": val_metrics['recall_macro']
                })
            
            # Save checkpoint
            is_best = val_metrics['accuracy'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics['accuracy']
                self.patience_counter = 0
                
                # Save best model
                self.save_checkpoint(
                    os.path.join("checkpoints", "best_model.pth"),
                    is_best=True
                )
                logger.info(f"New best model saved with validation accuracy: {self.best_val_acc:.4f}")
            else:
                self.patience_counter += 1
            
            # Save regular checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(
                    os.path.join("checkpoints", f"checkpoint_epoch_{epoch}.pth")
                )
            
            # Early stopping
            if self.patience_counter >= self.config["training"]["patience"]:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break
        
        logger.info("Training completed!")
    
    def save_checkpoint(self, filepath: str, is_best: bool = False):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_acc': self.best_val_acc,
            'config': self.config,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None
        }
        
        torch.save(checkpoint, filepath)
        
        if is_best and wandb.run:
            wandb.save(filepath)
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint['best_val_acc']
        
        logger.info(f"Checkpoint loaded from {filepath}")


def main():
    """Main training function"""
    
    # Load configuration
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Setup data processing
    data_processor = SparkDataProcessor(config)
    
    # Create or load sample data
    data_path = "data/processed"
    if not os.path.exists(f"{data_path}/train"):
        logger.info("Creating sample dataset...")
        sample_df = data_processor.create_sample_dataset("data/raw/sample.parquet", 1000)
        
        # Process data
        processed_df = data_processor.preprocess_images(sample_df)
        processed_df = data_processor.clean_text_data(processed_df)
        
        # Split data
        train_df, val_df, test_df = data_processor.create_train_val_test_split(processed_df)
        
        # Save processed data
        data_processor.save_processed_data(train_df, val_df, test_df, data_path)
    
    # Create dataloaders
    dataloaders = create_dataloaders(data_path, config)
    
    # Initialize trainer
    trainer = MultimodalTrainer(config)
    
    # Start training
    trainer.train(dataloaders['train'], dataloaders['val'])
    
    # Close Spark session
    data_processor.close()


if __name__ == "__main__":
    main()
