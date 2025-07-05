import torch
import numpy as np
import random
import os
from typing import Dict, Any
from loguru import logger


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")


def save_checkpoint(model, optimizer, scheduler, epoch, best_acc, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_acc': best_acc
    }
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(checkpoint, filepath)
    logger.info(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath, model, optimizer=None, scheduler=None, device='cpu'):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    best_acc = checkpoint.get('best_acc', 0.0)
    
    logger.info(f"Checkpoint loaded from {filepath}")
    return epoch, best_acc


def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device


def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    return total_params, trainable_params


def create_directories(paths):
    """Create directories if they don't exist"""
    for path in paths:
        os.makedirs(path, exist_ok=True)
        logger.info(f"Directory created/verified: {path}")


def setup_logging(config: Dict[str, Any]):
    """Setup logging configuration"""
    log_config = config.get("logging", {})
    log_level = log_config.get("level", "INFO")
    log_dir = log_config.get("log_dir", "./logs")
    
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logger
    logger.remove()  # Remove default handler
    logger.add(
        os.path.join(log_dir, "app.log"),
        rotation="10 MB",
        retention="7 days",
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
    )
    logger.add(
        lambda msg: print(msg, end=""),
        level=log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}"
    )
    
    logger.info("Logging setup completed")


class EarlyStopping:
    """Early stopping implementation"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif self._is_better(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop
    
    def _is_better(self, score: float) -> bool:
        if self.mode == 'max':
            return score > self.best_score + self.min_delta
        else:
            return score < self.best_score - self.min_delta


def freeze_layers(model, layer_names):
    """Freeze specific layers in the model"""
    frozen_count = 0
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in layer_names):
            param.requires_grad = False
            frozen_count += 1
    
    logger.info(f"Frozen {frozen_count} parameters")


def unfreeze_layers(model, layer_names):
    """Unfreeze specific layers in the model"""
    unfrozen_count = 0
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in layer_names):
            param.requires_grad = True
            unfrozen_count += 1
    
    logger.info(f"Unfrozen {unfrozen_count} parameters")


def get_model_size(model):
    """Get model size in MB"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    logger.info(f"Model size: {size_mb:.2f} MB")
    
    return size_mb


def warmup_learning_rate(optimizer, current_step, warmup_steps, base_lr):
    """Apply learning rate warmup"""
    if current_step < warmup_steps:
        lr = base_lr * (current_step + 1) / warmup_steps
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return optimizer.param_groups[0]['lr']


def cosine_annealing_lr(optimizer, current_epoch, max_epochs, base_lr, min_lr=0):
    """Apply cosine annealing learning rate schedule"""
    lr = min_lr + (base_lr - min_lr) * (1 + np.cos(np.pi * current_epoch / max_epochs)) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
