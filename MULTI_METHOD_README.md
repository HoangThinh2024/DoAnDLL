# Multi-Method Training System for Smart Pill Recognition

This enhanced system provides three different training methods for multimodal pill recognition: **PyTorch**, **PySpark**, and **HuggingFace Transformers**, with comprehensive comparison and benchmarking capabilities.

## 🚀 Features

### 🔥 PyTorch Training
- **Enhanced multimodal transformer** with Vision Transformer (ViT) + BERT
- **Cross-modal attention** for feature fusion
- **Mixed precision training** optimized for NVIDIA Quadro 6000
- **Advanced optimization** with AdamW, cosine annealing, gradient clipping
- **Comprehensive metrics** and early stopping

### ⚡ PySpark Training
- **Distributed training** for big data processing
- **Scalable feature extraction** from images and text
- **Multiple model types**: MLP, Random Forest
- **Hyperparameter tuning** with CrossValidator
- **Apache Spark integration** for handling large datasets

### 🤗 HuggingFace Transformers
- **State-of-the-art models** with latest architectures
- **Custom multimodal model** with HuggingFace Trainer API
- **Advanced callbacks** and training monitoring
- **Pre-trained model fine-tuning** for better performance
- **Easy model sharing** and deployment

### 📊 Comparison & Benchmarking
- **Unified model registry** for managing trained models
- **Comprehensive benchmarking** across all methods
- **Performance comparison** with detailed metrics
- **Visualization tools** for analysis
- **Export capabilities** for deployment

## 📁 Project Structure

```
DoAnDLL/
├── core/
│   ├── models/
│   │   ├── multimodal_transformer.py    # PyTorch multimodal model
│   │   └── model_registry.py            # Model management system
│   ├── training/
│   │   ├── trainer.py                   # Enhanced PyTorch trainer
│   │   ├── spark_trainer.py             # PySpark distributed trainer
│   │   ├── hf_trainer.py                # HuggingFace Transformers trainer
│   │   └── comparison.py                # Benchmarking framework
│   └── data/
│       └── data_processing.py           # Unified data processing
├── train_multi_method.py                # Main training script
├── demo_multi_method.py                 # Example usage demo
└── config/
    └── config.yaml                      # Configuration file
```

## 🛠️ Installation

### Prerequisites
- Python 3.10+
- CUDA 12.8+ (optional, for GPU acceleration)
- 16GB+ RAM recommended

### Quick Setup
```bash
# Clone repository
git clone <repository-url>
cd DoAnDLL

# Setup environment
./bin/pill-setup  # or make setup

# Activate environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
- **PyTorch**: 2.3+ with CUDA support
- **PySpark**: 3.5+ for distributed computing
- **Transformers**: 4.40+ for state-of-the-art models
- **Additional**: See `requirements.txt` for full list

## 🚀 Quick Start

### 1. Basic Training (Single Method)

```bash
# Train with PyTorch
python train_multi_method.py train --method pytorch --dataset ./data --model my_pytorch_model

# Train with PySpark
python train_multi_method.py train --method spark --dataset ./data --model my_spark_model

# Train with Transformers
python train_multi_method.py train --method transformers --dataset ./data --model my_hf_model
```

### 2. Train All Methods
```bash
# Train using all three methods
python train_multi_method.py train-all --dataset ./data --prefix experiment1
```

### 3. Run Benchmark
```bash
# Comprehensive benchmark
python train_multi_method.py benchmark --dataset ./data --output ./results

# Quick benchmark for testing
python train_multi_method.py benchmark --dataset ./data --output ./results --quick
```

### 4. Model Management
```bash
# List all models
python train_multi_method.py list-models

# List models by method
python train_multi_method.py list-models --method pytorch

# Load model for analysis
python train_multi_method.py load-model --model-id PT_my_model_20240101_120000

# Generate comparison report
python train_multi_method.py report
```

## 📊 Example Usage

### Demo Script
```bash
# Run comprehensive demo
python demo_multi_method.py
```

The demo showcases:
- Training with each method
- Model registry usage
- Comparison reports
- Benchmarking system

### Python API Usage

```python
from train_multi_method import MultiMethodTrainer
from core.models.model_registry import ModelRegistry

# Initialize trainer
trainer = MultiMethodTrainer('./config/config.yaml')

# Train single method
results = trainer.train_single_method(
    method='pytorch',
    dataset_path='./data',
    model_name='my_model'
)

# Train all methods
all_results = trainer.train_all_methods(
    dataset_path='./data',
    model_prefix='experiment1'
)

# Access model registry
registry = ModelRegistry()
models = registry.list_models()
best_model = registry.get_best_model(metric='accuracy')
```

## ⚙️ Configuration

The system uses a comprehensive YAML configuration file:

```yaml
# config/config.yaml
model:
  visual_encoder:
    model_name: "vit_base_patch16_224"
    pretrained: true
    output_dim: 768
  text_encoder:
    model_name: "bert-base-uncased"
    pretrained: true
    max_length: 128
  fusion:
    type: "cross_attention"
    num_attention_heads: 8
  classifier:
    num_classes: 1000
    hidden_dims: [512, 256]

training:
  batch_size: 32
  learning_rate: 1e-4
  num_epochs: 50
  optimizer: "adamw"
  scheduler: "cosine_annealing"

data:
  image_size: 224
  spark:
    app_name: "PillRecognitionSpark"
    executor_memory: "8g"
    driver_memory: "4g"

hardware:
  gpu:
    mixed_precision: true
    memory_fraction: 0.9
```

## 📈 Performance Comparison

| Method | Accuracy | Training Time | Memory Usage | Scalability |
|--------|----------|---------------|--------------|-------------|
| PyTorch | 95%+ | Medium | High | Medium |
| PySpark | 85%+ | High | Medium | Excellent |
| Transformers | 98%+ | Variable | High | Good |

### When to Use Each Method

**🔥 PyTorch**: 
- Best for: Custom architectures, research, fine-grained control
- Pros: Flexibility, performance, community support
- Cons: Requires more manual optimization

**⚡ PySpark**:
- Best for: Large datasets, distributed computing, traditional ML
- Pros: Scalability, built-in distributed processing
- Cons: Lower accuracy for complex vision tasks

**🤗 Transformers**:
- Best for: State-of-the-art results, production deployment
- Pros: Latest architectures, easy deployment, high accuracy
- Cons: Less customization, higher resource requirements

## 📊 Dataset Format

The system supports multiple dataset formats:

### JSON Format
```json
[
  {
    "image_path": "/path/to/image1.jpg",
    "text_imprint": "ADVIL 200",
    "label": 0,
    "class_name": "ibuprofen_200mg"
  }
]
```

### CSV Format
```csv
image_path,text_imprint,label,class_name
/path/to/image1.jpg,ADVIL 200,0,ibuprofen_200mg
/path/to/image2.jpg,TYLENOL 500,1,acetaminophen_500mg
```

### Directory Structure
```
dataset/
├── class1/
│   ├── image1.jpg
│   └── image2.jpg
└── class2/
    ├── image3.jpg
    └── image4.jpg
```

## 🔧 Advanced Features

### Model Registry
- **Centralized storage** for all trained models
- **Metadata tracking** (accuracy, training time, hyperparameters)
- **Version control** and model comparison
- **Export capabilities** for deployment

### Benchmarking System
- **Automated comparison** across methods
- **Statistical analysis** with confidence intervals
- **Visualization tools** for results
- **Comprehensive reporting**

### GPU Optimization
- **Mixed precision training** for faster training
- **Memory optimization** for NVIDIA Quadro 6000
- **Multi-GPU support** with DataParallel
- **CUDA 12.8 optimizations**

## 🚀 Deployment

### Model Export
```python
# Export best model for deployment
registry = ModelRegistry()
best_model = registry.get_best_model()
registry.export_model(best_model['model_id'], './deployment/')
```

### API Integration
```python
# Load model for inference
model_info = registry.load_model('model_id')
model = model_info['model']

# Make predictions
predictions = model.predict(images, texts)
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Make changes following the coding standards
4. Add tests for new functionality
5. Run full test suite: `python -m pytest`
6. Commit changes: `git commit -m 'Add amazing feature'`
7. Push to branch: `git push origin feature/amazing-feature`
8. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: Check this README and inline code documentation
- **Issues**: Open GitHub issues for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Examples**: Run `python demo_multi_method.py` for working examples

## 🙏 Acknowledgments

- **PyTorch Team** for the excellent deep learning framework
- **Apache Spark** community for distributed computing capabilities
- **HuggingFace** for state-of-the-art transformer models
- **DoAnDLL Team** for system design and implementation

---

**⭐ If this project helps you, please give it a star! ⭐**

*Made with ❤️ by DoAnDLL Team*