"""
Smart Pill Recognition System - Core Package

Core AI modules for pharmaceutical identification using multimodal transformers.
"""

__version__ = "1.0.0"
__author__ = "DoAnDLL Team"

# Import main components
try:
    from .models.multimodal_transformer import MultimodalPillTransformer
    from .models.model_registry import ModelRegistry, TrainingMethod
    from .data.data_processing import preprocess_image
    from .utils.utils import get_device
except ImportError:
    # Handle missing dependencies gracefully
    pass

__all__ = [
    'MultimodalPillTransformer',
    'ModelRegistry', 
    'TrainingMethod',
    'preprocess_image',
    'get_device'
]
