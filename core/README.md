# Smart Pill Recognition System - Core Package

Core AI modules for pharmaceutical identification using multimodal transformers.

## 📁 Module Structure

```
core/
├── __init__.py                    # Package initialization
├── models/                        # Neural network architectures
│   ├── __init__.py
│   ├── multimodal_transformer.py # Main multimodal model
│   └── model_registry.py         # Model registration system
├── data/                          # Data processing & datasets  
│   ├── __init__.py
│   ├── data_processing.py         # Image/text preprocessing
│   └── cure_dataset.py            # CURE dataset handler
├── training/                      # Training procedures
│   ├── __init__.py
│   ├── trainer.py                 # Standard PyTorch trainer
│   ├── spark_trainer.py           # Big data trainer
│   └── hf_trainer.py              # HuggingFace trainer
└── utils/                         # Utilities & helpers
    ├── __init__.py
    ├── utils.py                   # General utilities
    ├── metrics.py                 # Evaluation metrics
    └── port_manager.py            # Port management
```

## 🚀 Quick Usage

### Basic Model Loading

```python
from core.models.multimodal_transformer import MultimodalPillTransformer
from core.data.data_processing import preprocess_image

# Load model
model = MultimodalPillTransformer.load_pretrained()

# Process image
image_tensor = preprocess_image(image_path)

# Make prediction
result = model.predict(image_tensor, text_imprint="ADVIL 200")
print(f"Prediction: {result['class_name']} (confidence: {result['confidence']:.2%})")
```

### Custom Training

```python
from core.training.trainer import Trainer
from core.data.cure_dataset import CUREDataset

# Setup dataset
dataset = CUREDataset(data_path="./Dataset_BigData")

# Initialize trainer
trainer = Trainer(
    model=model,
    train_dataset=dataset.train,
    val_dataset=dataset.val,
    config="config/config.yaml"
)

# Start training
trainer.train()
```

### Big Data Processing with Spark

```python
from core.training.spark_trainer import SparkTrainer

# For large datasets (>100GB)
spark_trainer = SparkTrainer(
    data_path="./Dataset_BigData",
    config="config/config.yaml"
)

spark_trainer.train()
```

## 🧠 Model Architecture

The core implements a **multimodal transformer** that combines:

1. **Vision Transformer (ViT)**: For pill image analysis
2. **BERT**: For text imprint processing  
3. **Cross-Modal Attention**: For feature fusion
4. **Classification Head**: For final prediction

### Key Features

- ⚡ **High Performance**: 96.3% accuracy on CURE dataset
- 🚀 **Fast Inference**: 0.15s per image with GPU
- 🔧 **Flexible**: Supports multiple training methods
- 📊 **Scalable**: Handles large datasets with Spark

## 📊 Performance

| Metric | Value |
|--------|-------|
| Accuracy | 96.3% |
| Inference Speed | 0.15s/image |
| GPU Memory | ~3.2GB |
| Supported Classes | 1000+ |

## 🔧 Configuration

Models are configured through YAML files:

```yaml
model:
  vision:
    type: "vit_base_patch16_224"
    pretrained: true
  text:
    type: "bert-base-uncased"
    max_length: 128
  fusion:
    type: "cross_attention"
    num_heads: 8
  classifier:
    num_classes: 1000
```

## 🛠️ Development

### Adding New Models

1. Create model class in `models/`
2. Register in `model_registry.py`
3. Add configuration options
4. Implement training procedure

### Custom Data Processing

1. Extend `data_processing.py`
2. Add new augmentation methods
3. Create custom dataset classes
4. Update preprocessing pipeline

## 📚 API Reference

### Core Classes

- **`MultimodalPillTransformer`**: Main model class
- **`ModelRegistry`**: Model registration system
- **`CUREDataset`**: Dataset handling
- **`Trainer`**: Training orchestration

### Utility Functions

- **`preprocess_image()`**: Image preprocessing
- **`get_device()`**: GPU/CPU device detection
- **`load_checkpoint()`**: Model loading
- **`optimize_for_quadro_6000()`**: GPU optimization

For detailed API documentation, see the docstrings in each module.