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
        
        # Initialize mixed precision scaler
        self.use_amp = config.get("training", {}).get("use_amp", True)
        if self.use_amp:
            self.scaler = GradScaler()
        
        # Initialize metrics tracking
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)
        self.best_val_acc = 0.0
        self.current_epoch = 0
        
        # Initialize early stopping
        early_stop_config = config.get("training", {})
        self.early_stopping = EarlyStopping(
            patience=early_stop_config.get("patience", 10),
            min_delta=early_stop_config.get("min_delta", 0.001),
            monitor='val_accuracy'
        )
        
        # Setup checkpointing
        self.checkpoint_dir = Path(config.get("logging", {}).get("checkpoint_dir", "./checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Weights & Biases if configured
        wandb_config = config.get("logging", {}).get("wandb", {})
        if wandb_config.get("enabled", False):
            wandb.init(
                project=wandb_config.get("project", "pill-recognition"),
                entity=wandb_config.get("entity", None),
                config=config,
                name=wandb_config.get("run_name", None)
            )
            self.use_wandb = True
        else:
            self.use_wandb = False
        
        # Resume from checkpoint if specified
        if resume_from:
            self._load_checkpoint(resume_from)
        
        logger.info("Trainer initialized successfully")
    
    def _create_model(self):
        """Create and initialize model"""
        model = MultimodalPillTransformer(self.config["model"])
        
        if self.multi_gpu:
            model = nn.DataParallel(model)
        
        model.to(self.device)
        return model
    
    def _setup_optimizer_and_scheduler(self):
        """Setup optimizer and learning rate scheduler"""
        training_config = self.config["training"]
        
        # Optimizer
        optimizer_name = training_config.get("optimizer", "adamw").lower()
        lr = training_config["learning_rate"]
        weight_decay = training_config.get("weight_decay", 0.01)
        
        if optimizer_name == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif optimizer_name == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # Scheduler
        scheduler_name = training_config.get("scheduler", "cosine_annealing").lower()
        num_epochs = training_config["num_epochs"]
        
        if scheduler_name == "cosine_annealing":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=num_epochs,
                eta_min=lr * 0.01
            )
        elif scheduler_name == "step":
            self.scheduler = StepLR(
                self.optimizer,
                step_size=num_epochs // 3,
                gamma=0.1
            )
        elif scheduler_name == "onecycle":
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=lr,
                epochs=num_epochs,
                steps_per_epoch=100  # Will be updated with actual value
            )
        else:
            self.scheduler = None
        
        logger.info(f"Optimizer: {optimizer_name}, Scheduler: {scheduler_name}")
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = len(train_loader)
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            images = batch['image'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(images, input_ids, attention_mask)
                    loss = F.cross_entropy(outputs['logits'], labels)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.get("training", {}).get("gradient_clip_norm"):
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config["training"]["gradient_clip_norm"]
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images, input_ids, attention_mask)
                loss = F.cross_entropy(outputs['logits'], labels)
                
                loss.backward()
                
                # Gradient clipping
                if self.config.get("training", {}).get("gradient_clip_norm"):
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config["training"]["gradient_clip_norm"]
                    )
                
                self.optimizer.step()
            
            # Calculate accuracy
            predictions = torch.argmax(outputs['logits'], dim=-1)
            accuracy = (predictions == labels).float().mean()
            
            # Update metrics
            epoch_loss += loss.item()
            epoch_acc += accuracy.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{accuracy.item():.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            # Log to wandb
            if self.use_wandb and batch_idx % 100 == 0:
                wandb.log({
                    'train_batch_loss': loss.item(),
                    'train_batch_acc': accuracy.item(),
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'epoch': self.current_epoch
                })
        
        # Calculate average metrics
        avg_loss = epoch_loss / num_batches
        avg_acc = epoch_acc / num_batches
        
        # Store metrics
        self.train_metrics['loss'].append(avg_loss)
        self.train_metrics['accuracy'].append(avg_acc)
        
        return avg_loss, avg_acc
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        epoch_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        progress_bar = tqdm(val_loader, desc=f"Validation Epoch {self.current_epoch + 1}")
        
        with torch.no_grad():
            for batch in progress_bar:
                # Move data to device
                images = batch['image'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                if self.use_amp:
                    with autocast():
                        outputs = self.model(images, input_ids, attention_mask)
                        loss = F.cross_entropy(outputs['logits'], labels)
                else:
                    outputs = self.model(images, input_ids, attention_mask)
                    loss = F.cross_entropy(outputs['logits'], labels)
                
                # Get predictions and probabilities
                probabilities = F.softmax(outputs['logits'], dim=-1)
                predictions = torch.argmax(probabilities, dim=-1)
                
                # Store for metrics calculation
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                epoch_loss += loss.item()
        
        # Calculate comprehensive metrics
        avg_loss = epoch_loss / len(val_loader)
        metrics = self.metrics_calculator.calculate_metrics(all_labels, all_predictions)
        
        # Calculate top-k accuracy
        top5_acc = self.metrics_calculator.calculate_top_k_accuracy(
            all_labels, np.array(all_probabilities), k=5
        )
        metrics['top5_accuracy'] = top5_acc
        
        # Store metrics
        self.val_metrics['loss'].append(avg_loss)
        for key, value in metrics.items():
            self.val_metrics[key].append(value)
        
        return avg_loss, metrics
    
    def train(self, train_loader, val_loader, num_epochs: Optional[int] = None):
        """Complete training loop"""
        if num_epochs is None:
            num_epochs = self.config["training"]["num_epochs"]
        
        logger.info(f"Starting training for {num_epochs} epochs")
        start_time = time.time()
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_metrics = self.validate_epoch(val_loader)
            val_acc = val_metrics['accuracy']
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, OneCycleLR):
                    # OneCycleLR is updated per batch, not per epoch
                    pass
                else:
                    self.scheduler.step()
            
            # Log epoch results
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                f"Top-5 Acc: {val_metrics['top5_accuracy']:.4f}"
            )
            
            # Log to wandb
            if self.use_wandb:
                log_dict = {
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'train_accuracy': train_acc,
                    'val_loss': val_loss,
                }
                log_dict.update({f'val_{k}': v for k, v in val_metrics.items()})
                wandb.log(log_dict)
            
            # Save checkpoint if best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self._save_checkpoint(epoch, is_best=True)
                logger.info(f"New best model saved with validation accuracy: {val_acc:.4f}")
            
            # Regular checkpoint saving
            if (epoch + 1) % self.config.get("training", {}).get("save_every", 10) == 0:
                self._save_checkpoint(epoch, is_best=False)
            
            # Early stopping check
            if self.early_stopping(val_acc):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")
        
        # Final evaluation
        self._final_evaluation(val_loader)
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        model_state = self.model.module.state_dict() if self.multi_gpu else self.model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_acc': self.best_val_acc,
            'train_metrics': dict(self.train_metrics),
            'val_metrics': dict(self.val_metrics),
            'config': self.config
        }
        
        if is_best:
            filepath = self.checkpoint_dir / "best_model.pth"
        else:
            filepath = self.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pth"
        
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved: {filepath}")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint for resuming training"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        if self.multi_gpu:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.current_epoch = checkpoint['epoch'] + 1
        self.best_val_acc = checkpoint['best_val_acc']
        self.train_metrics = defaultdict(list, checkpoint.get('train_metrics', {}))
        self.val_metrics = defaultdict(list, checkpoint.get('val_metrics', {}))
        
        logger.info(f"Resumed training from epoch {self.current_epoch}")
    
    def _final_evaluation(self, val_loader):
        """Perform final comprehensive evaluation"""
        logger.info("Performing final evaluation...")
        
        # Load best model
        best_model_path = self.checkpoint_dir / "best_model.pth"
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location=self.device)
            if self.multi_gpu:
                self.model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Detailed evaluation
        val_loss, val_metrics = self.validate_epoch(val_loader)
        
        # Generate and save detailed report
        self._save_evaluation_report(val_metrics)
        
        logger.info("Final evaluation completed")
    
    def _save_evaluation_report(self, metrics):
        """Save detailed evaluation report"""
        report = {
            'best_validation_accuracy': self.best_val_acc,
            'final_metrics': metrics,
            'training_history': {
                'train_metrics': dict(self.train_metrics),
                'val_metrics': dict(self.val_metrics)
            },
            'config': self.config
        }
        
        report_path = self.checkpoint_dir / "evaluation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Evaluation report saved: {report_path}")


class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, monitor: str = 'val_accuracy'):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.best_score = None
        self.counter = 0
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.should_stop
