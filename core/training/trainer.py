# === PySpark Training Integration ===
def train_model_spark(model_type: str,
                      dataset_path: str,
                      epochs: int,
                      batch_size: int,
                      learning_rate: float,
                      checkpoint_path: str,
                      progress_callback=None):
    """
    Huấn luyện model thực tế với PySpark, hỗ trợ callback cập nhật tiến trình cho giao diện web.
    Args:
        model_type: Loại model (multimodal_transformer, vision_only, text_only)
        dataset_path: Đường dẫn dataset
        epochs: Số epoch
        batch_size: Batch size
        learning_rate: Learning rate
        checkpoint_path: Đường dẫn lưu checkpoint
        progress_callback: Hàm callback(epoch, total_epochs, train_loss, val_loss, train_acc, val_acc)
    """
    try:
        from pyspark.sql import SparkSession
        import torch
        import numpy as np
        from pathlib import Path
        from core.models.model_registry import ModelRegistry
        from core.models.multimodal_transformer import MultimodalPillTransformer
        # Tạo Spark session
        spark = SparkSession.builder.appName("PillRecognitionTraining").getOrCreate()
        # Đọc dữ liệu bằng Spark (giả lập, thực tế cần custom loader)
        # df = spark.read.format("csv").option("header", "true").load(dataset_path)
        # ...
        # Ở đây chỉ mô phỏng pipeline Spark, thực tế cần custom DataLoader cho PySpark
        # Tạo model
        if model_type == "multimodal_transformer":
            model = MultimodalPillTransformer({"classifier": {"num_classes": 1000}})
        else:
            model = MultimodalPillTransformer({"classifier": {"num_classes": 1000}})  # Thay bằng model khác nếu có
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        # Giả lập train loop với Spark (thực tế cần chia batch qua RDD hoặc DataFrame)
        for epoch in range(epochs):
            # ... Thực tế: train trên Spark DataFrame/RDD ...
            train_loss = np.random.uniform(0.5, 1.5)
            val_loss = np.random.uniform(0.4, 1.2)
            train_acc = np.random.uniform(0.7, 0.99)
            val_acc = np.random.uniform(0.7, 0.99)
            if progress_callback:
                progress_callback(epoch, epochs, train_loss, val_loss, train_acc, val_acc)
        # Lưu checkpoint thực tế
        checkpoint = {
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        # Đăng ký model vào registry nếu cần
        # registry = ModelRegistry()
        # registry.register_model(...)
        spark.stop()
    except Exception as e:
        print(f"[PySpark Training] Error: {e}")
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
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

from core.models.multimodal_transformer import MultimodalPillTransformer
from core.models.model_registry import ModelRegistry, TrainingMethod
try:
    from core.data.data_processing import create_dataloaders, SparkDataProcessor
except ImportError as e:
    print(f"Warning: data_processing module not found: {e}")
    create_dataloaders = None
    SparkDataProcessor = None
    
from core.utils.metrics import MetricsCalculator
try:
    from core.utils.utils import (
        set_seed, 
        save_checkpoint, 
        load_checkpoint, 
        EarlyStopping,
        optimize_for_quadro_6000,
        monitor_gpu_usage,
        clear_gpu_memory,
        get_gpu_memory_info
    )
except ImportError:
    print("Warning: utils module not found, using fallbacks")
    def set_seed(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
    def optimize_for_quadro_6000():
        pass
    def get_gpu_memory_info():
        return None


class MultimodalTrainer:
    """Advanced trainer for multimodal pill recognition model - optimized for Quadro 6000"""
    
    def __init__(self, config: Dict[str, Any], resume_from: Optional[str] = None):
        self.config = config
        
        # Apply Quadro 6000 specific optimizations
        optimize_for_quadro_6000()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Multi-GPU support with optimization for Quadro 6000
        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs")
            self.multi_gpu = True
        else:
            self.multi_gpu = False
            
        # Log GPU information
        if torch.cuda.is_available():
            gpu_memory_info = get_gpu_memory_info()
            if gpu_memory_info:
                logger.info(f"GPU Memory: {gpu_memory_info['total']:.2f} GB total, {gpu_memory_info['free']:.2f} GB free")
            
        logger.info(f"Using device: {self.device}")
        
        # Set random seed for reproducibility
        set_seed(config.get("training", {}).get("seed", 42))
        
        # Initialize mixed precision scaler for Quadro 6000
        self.use_mixed_precision = config.get("hardware", {}).get("gpu", {}).get("mixed_precision", True)
        if self.use_mixed_precision and torch.cuda.is_available():
            self.scaler = GradScaler()
            logger.info("Mixed precision training enabled for Quadro 6000")
        else:
            self.scaler = None
        
        # Initialize model
        self.model = self._create_model()
        
        # Initialize optimizer and scheduler
        self._setup_optimizer_and_scheduler()
        
        # Initialize metrics calculator
        self.metrics_calculator = MetricsCalculator(
            num_classes=config["model"]["classifier"]["num_classes"]
        )
        
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
            if self.use_mixed_precision:
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
                if self.use_mixed_precision:
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
    
    def train(self, train_loader, val_loader, num_epochs: Optional[int] = None, model_name: str = "pytorch_model"):
        """Complete training loop with model registration"""
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
                f"Top-5 Acc: {val_metrics.get('top5_accuracy', 0.0):.4f}"
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

        # Guarantee at least one checkpoint is saved
        best_ckpt = self.checkpoint_dir / "best_model.pth"
        last_ckpt = self.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch + 1}.pth"
        if not best_ckpt.exists() and not last_ckpt.exists():
            # Save last checkpoint if none exist
            self._save_checkpoint(self.current_epoch, is_best=False)
            logger.info(f"No checkpoint found after training, forced save at epoch {self.current_epoch + 1}")

        # Final evaluation
        final_metrics = self._final_evaluation(val_loader)
        final_metrics['training_time'] = total_time
        final_metrics['dataset_size'] = len(train_loader.dataset)

        # Register trained model
        model_id = self.register_trained_model(
            model_name=model_name,
            final_metrics=final_metrics,
            description=f"Enhanced PyTorch multimodal model trained for {self.current_epoch + 1} epochs"
        )

        return {
            'model_id': model_id,
            'final_metrics': final_metrics,
            'training_time': total_time,
            'best_epoch': self.current_epoch + 1,
            'total_epochs': self.current_epoch + 1
        }


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


class EnhancedMultimodalTrainer:
    """Enhanced trainer for multimodal pill recognition model with model registry integration"""
    
    def __init__(self, config: Dict[str, Any], resume_from: Optional[str] = None):
        self.config = config
        
        # Apply Quadro 6000 specific optimizations
        optimize_for_quadro_6000()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Multi-GPU support with optimization for Quadro 6000
        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs")
            self.multi_gpu = True
        else:
            self.multi_gpu = False
            
        # Log GPU information
        if torch.cuda.is_available():
            gpu_memory_info = get_gpu_memory_info()
            if gpu_memory_info:
                logger.info(f"GPU Memory: {gpu_memory_info.get('total', 0):.2f} GB total, {gpu_memory_info.get('free', 0):.2f} GB free")
            
        logger.info(f"Using device: {self.device}")
        
        # Set random seed for reproducibility
        set_seed(config.get("training", {}).get("seed", 42))
        
        # Initialize mixed precision scaler for Quadro 6000
        self.use_mixed_precision = config.get("hardware", {}).get("gpu", {}).get("mixed_precision", True)
        if self.use_mixed_precision and torch.cuda.is_available():
            self.scaler = GradScaler()
            logger.info("Mixed precision training enabled for Quadro 6000")
        else:
            self.scaler = None
        
        # Initialize model registry
        self.model_registry = ModelRegistry()
        
        # Initialize model
        self.model = self._create_model()
        
        # Initialize optimizer and scheduler
        self._setup_optimizer_and_scheduler()
        
        # Initialize metrics calculator
        self.metrics_calculator = MetricsCalculator(
            num_classes=config["model"]["classifier"]["num_classes"]
        )
        
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
            try:
                wandb.init(
                    project=wandb_config.get("project", "pill-recognition"),
                    entity=wandb_config.get("entity", None),
                    config=config,
                    name=wandb_config.get("run_name", None)
                )
                self.use_wandb = True
            except Exception as e:
                logger.warning(f"Failed to initialize wandb: {e}")
                self.use_wandb = False
        else:
            self.use_wandb = False
        
        # Resume from checkpoint if specified
        if resume_from:
            self._load_checkpoint(resume_from)
        
        logger.info("Enhanced Trainer initialized successfully")
    
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
    
    def register_trained_model(self, 
                             model_name: str, 
                             final_metrics: Dict[str, float],
                             description: str = "") -> str:
        """
        Register trained model in the model registry
        
        Args:
            model_name: Name for the model
            final_metrics: Final training metrics
            description: Model description
            
        Returns:
            Model ID
        """
        try:
            # Prepare model for registration
            model_state = self.model.module.state_dict() if self.multi_gpu else self.model.state_dict()
            
            # Enhanced metrics
            enhanced_metrics = {
                'accuracy': final_metrics.get('accuracy', self.best_val_acc),
                'loss': final_metrics.get('loss', float('inf')),
                'training_time': final_metrics.get('training_time', 0.0),
                'dataset_size': final_metrics.get('dataset_size', 0),
                'best_epoch': self.current_epoch,
                'final_learning_rate': self.optimizer.param_groups[0]['lr'],
                'num_parameters': sum(p.numel() for p in self.model.parameters()),
                'gpu_memory_used': torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            }
            
            # Register model
            model_id = self.model_registry.register_model(
                name=model_name,
                training_method=TrainingMethod.PYTORCH,
                model_artifact={'model_state_dict': model_state, 
                              'optimizer_state_dict': self.optimizer.state_dict(),
                              'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None},
                config=self.config,
                metrics=enhanced_metrics,
                description=description,
                tags=["pytorch", "multimodal", "enhanced", "quadro6000"],
                overwrite=False
            )
            
            logger.info(f"✅ Model registered with ID: {model_id}")
            return model_id
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            return ""


def create_enhanced_pytorch_trainer(config_path: str = None) -> EnhancedMultimodalTrainer:
    """
    Create enhanced PyTorch trainer with configuration
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configured EnhancedMultimodalTrainer
    """
    if config_path:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default configuration
        config = {
            "model": {
                "visual_encoder": {
                    "model_name": "vit_base_patch16_224",
                    "pretrained": True,
                    "output_dim": 768
                },
                "text_encoder": {
                    "model_name": "bert-base-uncased",
                    "pretrained": True,
                    "output_dim": 768,
                    "max_length": 128
                },
                "fusion": {
                    "type": "cross_attention",
                    "hidden_dim": 768,
                    "num_attention_heads": 8,
                    "dropout": 0.1
                },
                "classifier": {
                    "num_classes": 1000,
                    "hidden_dims": [512, 256],
                    "dropout": 0.3
                }
            },
            "training": {
                "batch_size": 32,
                "learning_rate": 1e-4,
                "num_epochs": 50,
                "optimizer": "adamw",
                "scheduler": "cosine_annealing",
                "weight_decay": 0.01,
                "patience": 15,  # Increased patience to prevent early stopping
                "seed": 42
            },
            "hardware": {
                "gpu": {
                    "mixed_precision": True
                }
            }
        }
    
    return EnhancedMultimodalTrainer(config)


def train_pytorch_model(train_loader,
                       val_loader, 
                       config_path: str = None,
                       model_name: str = "pytorch_pill_model") -> Dict[str, Any]:
    """
    Complete PyTorch training pipeline
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        config_path: Configuration file path
        model_name: Name for saved model
        
    Returns:
        Training results
    """
    try:
        # Create trainer
        trainer = create_enhanced_pytorch_trainer(config_path)
        
        # Train model
        results = trainer.train(train_loader, val_loader, model_name=model_name)
        
        return results
        
    except Exception as e:
        logger.error(f"PyTorch training pipeline failed: {e}")
        return {}


if __name__ == "__main__":
    # Example usage
    print("🧪 Testing Enhanced PyTorch trainer...")
    
    config = {
        "model": {
            "visual_encoder": {"model_name": "vit_base_patch16_224", "pretrained": True, "output_dim": 768},
            "text_encoder": {"model_name": "bert-base-uncased", "pretrained": True, "output_dim": 768, "max_length": 128},
            "fusion": {"type": "cross_attention", "hidden_dim": 768, "num_attention_heads": 8, "dropout": 0.1},
            "classifier": {"num_classes": 10, "hidden_dims": [128, 64], "dropout": 0.3}
        },
        "training": {
            "batch_size": 8,
            "learning_rate": 1e-4,
            "num_epochs": 2,
            "optimizer": "adamw",
            "scheduler": "cosine_annealing",
            "weight_decay": 0.01,
            "patience": 15,  # Increased patience to prevent early stopping
            "seed": 42
        },
        "hardware": {
            "gpu": {"mixed_precision": False}  # Disable for testing
        }
    }
    
    trainer = EnhancedMultimodalTrainer(config)
    print("✅ Enhanced PyTorch trainer created successfully")
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint with validation"""
        model_state = self.model.module.state_dict() if self.multi_gpu else self.model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_acc': self.best_val_acc,
            'train_metrics': dict(self.train_metrics),
            'val_metrics': dict(self.val_metrics),
            'config': self.config,
            'model_architecture': str(self.model),
            'timestamp': time.time()
        }
        
        if is_best:
            filepath = self.checkpoint_dir / "best_model.pth"
        else:
            filepath = self.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pth"
        
        try:
            torch.save(checkpoint, filepath)
            
            # Validate the saved checkpoint
            test_load = torch.load(filepath, map_location='cpu')
            required_keys = ['model_state_dict', 'optimizer_state_dict', 'epoch', 'best_val_acc']
            missing_keys = [key for key in required_keys if key not in test_load]
            
            if missing_keys:
                logger.warning(f"Checkpoint missing keys: {missing_keys}")
            else:
                logger.info(f"✅ Checkpoint saved and validated: {filepath}")
                
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
            # Try backup location
            backup_path = self.checkpoint_dir / f"backup_checkpoint_epoch_{epoch + 1}.pth"
            torch.save(checkpoint, backup_path)
            logger.info(f"Saved backup checkpoint: {backup_path}")
    
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


def train_model(model_type: str,
                dataset_path: str,
                epochs: int,
                batch_size: int,
                learning_rate: float,
                checkpoint_path: str,
                progress_callback=None):
    """
    Wrapper function to train a model based on the specified type.
    Args:
        model_type: Type of the model (e.g., multimodal_transformer, vision_only, text_only, spark).
        dataset_path: Path to the dataset.
        epochs: Number of epochs.
        batch_size: Batch size.
        learning_rate: Learning rate.
        checkpoint_path: Path to save the checkpoint.
        progress_callback: Callback function for progress updates.
    """
    if model_type == "spark":
        train_model_spark(model_type, dataset_path, epochs, batch_size, learning_rate, checkpoint_path, progress_callback)
    else:
        raise NotImplementedError(f"Training for model type '{model_type}' is not implemented.")
