import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, StepLR
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import wandb
from loguru import logger
from typing import Dict, Any, Optional, Tuple, List
import yaml
import json
from pathlib import Path
import time
from collections import defaultdict

from ..models.multimodal_transformer import MultimodalPillTransformer
from ..data.data_processing import create_dataloaders, SparkDataProcessor
from ..utils.metrics import MetricsCalculator
from ..utils.utils import set_seed, save_checkpoint, load_checkpoint, EarlyStopping


class MultimodalTrainer:
    """Advanced trainer for multimodal pill recognition model"""
    
    def __init__(self, config: Dict[str, Any], resume_from: Optional[str] = None):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Multi-GPU support
        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs")
            self.multi_gpu = True
        else:
            self.multi_gpu = False
            
        logger.info(f"Using device: {self.device}")
        
        # Set random seed for reproducibility
        set_seed(config.get("training", {}).get("seed", 42))
        
        # Initialize model
        self.model = self._create_model()
        
        # Initialize optimizer and scheduler
        self._setup_optimizer_and_scheduler()
        
        # Initialize metrics calculator
        self.metrics_calculator = MetricsCalculator(
            num_classes=config["model"]["classifier"]["num_classes"]
        )
        
        # Mixed precision training
        self.use_amp = config.get("training", {}).get("use_amp", True) and self.device.type == "cuda"
        self.scaler = GradScaler() if self.use_amp else None
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config["training"]["early_stopping"]["patience"],
            min_delta=config["training"]["early_stopping"]["min_delta"]
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.global_step = 0
        self.train_losses = []
        self.val_losses = []
        
        # Experiment tracking
        self.use_wandb = config.get("logging", {}).get("use_wandb", False)
        if self.use_wandb:
            self._setup_wandb()
    
    def _create_model(self) -> nn.Module:
        """Create and initialize the model"""
        model = MultimodalPillTransformer(self.config["model"])
        
        # Multi-GPU support
        if self.multi_gpu:
            model = nn.DataParallel(model)
        
        model = model.to(self.device)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Model created with {total_params:,} total parameters")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        return model
    
    def _setup_optimizer_and_scheduler(self):
        """Setup optimizer and learning rate scheduler"""
        
        # Get model parameters (handle DataParallel)
        model_params = self.model.module.parameters() if self.multi_gpu else self.model.parameters()
        
        # Optimizer
        optimizer_config = self.config["training"]["optimizer"]
        if optimizer_config["name"].lower() == "adamw":
            self.optimizer = optim.AdamW(
                model_params,
                lr=optimizer_config["learning_rate"],
                weight_decay=optimizer_config["weight_decay"],
                betas=optimizer_config.get("betas", (0.9, 0.999))
            )
        elif optimizer_config["name"].lower() == "adam":
            self.optimizer = optim.Adam(
                model_params,
                lr=optimizer_config["learning_rate"],
                weight_decay=optimizer_config["weight_decay"]
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_config['name']}")
        
        # Scheduler
        scheduler_config = self.config["training"]["scheduler"]
        if scheduler_config["name"] == "cosine_annealing":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config["training"]["num_epochs"],
                eta_min=scheduler_config.get("eta_min", 1e-7)
            )
        elif scheduler_config["name"] == "onecycle":
            total_steps = self.config["training"]["num_epochs"] * scheduler_config.get("steps_per_epoch", 100)
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=optimizer_config["learning_rate"],
                total_steps=total_steps,
                pct_start=scheduler_config.get("pct_start", 0.3)
            )
        elif scheduler_config["name"] == "step":
            self.scheduler = StepLR(
                self.optimizer,
                step_size=scheduler_config.get("step_size", 10),
                gamma=scheduler_config.get("gamma", 0.1)
            )
        else:
            self.scheduler = None
            
        logger.info(f"Optimizer: {optimizer_config['name']}, Scheduler: {scheduler_config.get('name', 'None')}")
    
    def _setup_wandb(self):
        """Setup Weights & Biases logging"""
        wandb_config = self.config["logging"]["wandb"]
        
        wandb.init(
            project=wandb_config["project"],
            entity=wandb_config.get("entity"),
            name=wandb_config.get("run_name", f"multimodal_pill_{int(time.time())}"),
            config=self.config,
            tags=wandb_config.get("tags", [])
        )
        
        # Watch model
        wandb.watch(self.model, log="all", log_freq=wandb_config.get("log_freq", 100))
        
    def _resume_training(self, checkpoint_path: str):
        """Resume training from checkpoint"""
        logger.info(f"Resuming training from {checkpoint_path}")
        
        checkpoint = load_checkpoint(checkpoint_path, self.model, self.optimizer, self.scheduler)
        
        self.current_epoch = checkpoint["epoch"]
        self.best_val_acc = checkpoint["best_val_acc"]
        self.global_step = checkpoint["global_step"]
        
        logger.info(f"Resumed from epoch {self.current_epoch}, best val acc: {self.best_val_acc:.4f}")
    
    
    def train_epoch(self, dataloader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        running_loss = 0.0
        all_predictions = []
        all_labels = []
        
        # Get tokenizer from model
        tokenizer = self.model.get_text_tokenizer() if not self.multi_gpu else self.model.module.get_text_tokenizer()
        
        pbar = tqdm(dataloader, desc=f"Training Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            images = batch['images'].to(self.device, non_blocking=True)
            texts = batch['texts']
            labels = batch['labels'].to(self.device, non_blocking=True)
            
            # Tokenize texts
            text_inputs = tokenizer(
                texts,
                max_length=self.config["model"]["text_encoder"]["max_length"],
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            # Move text inputs to device
            for key in text_inputs:
                text_inputs[key] = text_inputs[key].to(self.device, non_blocking=True)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(images, text_inputs)
                    loss = nn.CrossEntropyLoss()(outputs, labels)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config["training"].get("gradient_clip_norm"):
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config["training"]["gradient_clip_norm"]
                    )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images, text_inputs)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config["training"].get("gradient_clip_norm"):
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config["training"]["gradient_clip_norm"]
                    )
                
                # Optimizer step
                self.optimizer.step()
            
            # Update scheduler (if OneCycle)
            if self.scheduler and isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()
            
            # Update metrics
            running_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{running_loss / (batch_idx + 1):.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # Wandb logging
            if self.use_wandb and self.global_step % self.config["logging"]["log_interval"] == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                    'global_step': self.global_step
                })
            
            self.global_step += 1
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = np.mean(np.array(all_predictions) == np.array(all_labels))
        
        # Update scheduler (if not OneCycle)
        if self.scheduler and not isinstance(self.scheduler, OneCycleLR):
            self.scheduler.step()
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc
        }
    
    def validate_epoch(self, dataloader) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        
        running_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        # Get tokenizer from model
        tokenizer = self.model.get_text_tokenizer() if not self.multi_gpu else self.model.module.get_text_tokenizer()
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc=f"Validation Epoch {self.current_epoch + 1}")
            
            for batch_idx, batch in enumerate(pbar):
                # Move data to device
                images = batch['images'].to(self.device, non_blocking=True)
                texts = batch['texts']
                labels = batch['labels'].to(self.device, non_blocking=True)
                
                # Tokenize texts
                text_inputs = tokenizer(
                    texts,
                    max_length=self.config["model"]["text_encoder"]["max_length"],
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                )
                
                # Move text inputs to device
                for key in text_inputs:
                    text_inputs[key] = text_inputs[key].to(self.device, non_blocking=True)
                
                # Forward pass
                if self.use_amp:
                    with autocast():
                        outputs = self.model(images, text_inputs)
                        loss = nn.CrossEntropyLoss()(outputs, labels)
                else:
                    outputs = self.model(images, text_inputs)
                    loss = nn.CrossEntropyLoss()(outputs, labels)
                
                # Update metrics
                running_loss += loss.item()
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Avg Loss': f'{running_loss / (batch_idx + 1):.4f}'
                })
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(dataloader)
        
        # Calculate detailed metrics
        metrics = self.metrics_calculator.calculate_metrics(
            all_labels, all_predictions, all_probabilities
        )
        
        return {
            'loss': epoch_loss,
            **metrics
        }
    
    def train(self, train_dataloader, val_dataloader, test_dataloader=None):
        """Main training loop"""
        logger.info("Starting training...")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config["training"]["num_epochs"]):
            self.current_epoch = epoch
            
            # Training phase
            train_metrics = self.train_epoch(train_dataloader)
            self.train_losses.append(train_metrics['loss'])
            
            # Validation phase
            val_metrics = self.validate_epoch(val_dataloader)
            self.val_losses.append(val_metrics['loss'])
            
            # Logging
            logger.info(f"Epoch {epoch + 1}/{self.config['training']['num_epochs']}")
            logger.info(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
            
            # Wandb logging
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train/epoch_loss': train_metrics['loss'],
                    'train/epoch_accuracy': train_metrics['accuracy'],
                    'val/epoch_loss': val_metrics['loss'],
                    'val/epoch_accuracy': val_metrics['accuracy'],
                    'val/f1_score': val_metrics.get('f1_score', 0),
                    'val/precision': val_metrics.get('precision', 0),
                    'val/recall': val_metrics.get('recall', 0)
                })
            
            # Save checkpoint
            is_best = val_metrics['accuracy'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics['accuracy']
                logger.info(f"New best validation accuracy: {self.best_val_acc:.4f}")
            
            # Save checkpoint
            checkpoint_path = self._save_checkpoint(is_best, val_metrics)
            
            # Early stopping
            if self.early_stopping(val_metrics['loss']):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        # Final evaluation on test set
        if test_dataloader:
            logger.info("Evaluating on test set...")
            # Load best model
            best_checkpoint = os.path.join(
                self.config["training"]["checkpoint_dir"], 
                "best_model.pth"
            )
            if os.path.exists(best_checkpoint):
                self._load_checkpoint(best_checkpoint)
            
            test_metrics = self.validate_epoch(test_dataloader)
            logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
            logger.info(f"Test F1 Score: {test_metrics.get('f1_score', 0):.4f}")
            
            if self.use_wandb:
                wandb.log({
                    'test/accuracy': test_metrics['accuracy'],
                    'test/f1_score': test_metrics.get('f1_score', 0),
                    'test/precision': test_metrics.get('precision', 0),
                    'test/recall': test_metrics.get('recall', 0)
                })
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time / 3600:.2f} hours")
        
        if self.use_wandb:
            wandb.finish()
    
    def _save_checkpoint(self, is_best: bool, metrics: Dict[str, float]) -> str:
        """Save model checkpoint"""
        checkpoint_dir = Path(self.config["training"]["checkpoint_dir"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Model state (handle DataParallel)
        model_state = self.model.module.state_dict() if self.multi_gpu else self.model.state_dict()
        
        checkpoint = {
            'epoch': self.current_epoch + 1,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_acc': self.best_val_acc,
            'global_step': self.global_step,
            'config': self.config,
            'metrics': metrics
        }
        
        # Save latest checkpoint
        latest_path = checkpoint_dir / "latest_checkpoint.pth"
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved to {best_path}")
        
        # Save epoch checkpoint
        epoch_path = checkpoint_dir / f"checkpoint_epoch_{self.current_epoch + 1}.pth"
        torch.save(checkpoint, epoch_path)
        
        return str(latest_path)
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state (handle DataParallel)
        if self.multi_gpu:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer and scheduler
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.current_epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint['best_val_acc']
        self.global_step = checkpoint['global_step']
        
        logger.info(f"Checkpoint loaded: epoch {self.current_epoch}, best val acc: {self.best_val_acc:.4f}")
    
    def predict(self, dataloader) -> Tuple[List[int], List[float]]:
        """Make predictions on a dataset"""
        self.model.eval()
        
        all_predictions = []
        all_probabilities = []
        
        # Get tokenizer from model
        tokenizer = self.model.get_text_tokenizer() if not self.multi_gpu else self.model.module.get_text_tokenizer()
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                # Move data to device
                images = batch['images'].to(self.device, non_blocking=True)
                texts = batch['texts']
                
                # Tokenize texts
                text_inputs = tokenizer(
                    texts,
                    max_length=self.config["model"]["text_encoder"]["max_length"],
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                )
                
                # Move text inputs to device
                for key in text_inputs:
                    text_inputs[key] = text_inputs[key].to(self.device, non_blocking=True)
                
                # Forward pass
                if self.use_amp:
                    with autocast():
                        outputs = self.model(images, text_inputs)
                else:
                    outputs = self.model(images, text_inputs)
                
                # Get predictions and probabilities
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return all_predictions, all_probabilities


def train_model(config_path: str, resume_from: Optional[str] = None):
    """Main training function"""
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup data processing
    spark_processor = SparkDataProcessor(config)
    
    # Create sample dataset if it doesn't exist
    processed_data_path = config["data"]["processed_data_path"]
    if not os.path.exists(processed_data_path):
        logger.info("Creating sample dataset...")
        raw_data_path = config["data"]["raw_data_path"]
        os.makedirs(raw_data_path, exist_ok=True)
        
        # Create sample data
        sample_df = spark_processor.create_sample_dataset(
            output_path=os.path.join(raw_data_path, "sample_data"),
            num_samples=config["data"]["sample_size"]
        )
        
        # Process data
        df_processed = spark_processor.preprocess_images(sample_df)
        df_cleaned = spark_processor.clean_text_data(df_processed)
        
        # Split data
        train_df, val_df, test_df = spark_processor.create_train_val_test_split(df_cleaned)
        
        # Save processed data
        spark_processor.save_processed_data(train_df, val_df, test_df, processed_data_path)
        
        # Close Spark session
        spark_processor.close()
    
    # Create dataloaders
    dataloaders = create_dataloaders(processed_data_path, config)
    
    # Initialize trainer
    trainer = MultimodalTrainer(config, resume_from)
    
    # Start training
    trainer.train(
        train_dataloader=dataloaders['train'],
        val_dataloader=dataloaders['val'],
        test_dataloader=dataloaders['test']
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Multimodal Pill Recognition Model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    train_model(args.config, args.resume)
