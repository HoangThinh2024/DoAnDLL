# 🤝 Contributing to Smart Pill Recognition System

Thank you for your interest in contributing to the Smart Pill Recognition System! This document provides guidelines for contributing to the project.

## 📋 Table of Contents

- [🚀 Quick Start for Contributors](#-quick-start-for-contributors)
- [🛠️ Development Setup](#️-development-setup)
- [📝 Code Guidelines](#-code-guidelines)
- [🧪 Testing](#-testing)
- [📚 Documentation](#-documentation)
- [🔄 Pull Request Process](#-pull-request-process)
- [🐛 Reporting Issues](#-reporting-issues)

## 🚀 Quick Start for Contributors

```bash
# 1. Fork the repository on GitHub
# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/DoAnDLL.git
cd DoAnDLL

# 3. Set up development environment
./bin/pill-setup --dev
source .venv/bin/activate

# 4. Install development dependencies
uv pip install -r requirements-dev.txt

# 5. Create a feature branch
git checkout -b feature/your-feature-name

# 6. Make your changes and commit
git add .
git commit -m "Add your feature"

# 7. Push and create pull request
git push origin feature/your-feature-name
```

## 🛠️ Development Setup

### Prerequisites

- Python 3.10+
- Git
- UV package manager (recommended)

### Full Development Environment

```bash
# Install development tools
uv pip install pytest black isort flake8 mypy pre-commit jupyter

# Install pre-commit hooks
pre-commit install

# Validate setup
python tools/validate_setup.py
```

### Development Workflow

```bash
# Start development
source .venv/bin/activate

# Run tests before making changes
pytest tests/ -v

# Make your changes...

# Format code
black . --line-length 100
isort . --profile black

# Type checking
mypy core/ --ignore-missing-imports

# Lint
flake8 core/ --max-line-length 100

# Run tests after changes
pytest tests/ -v --cov=core

# Commit changes
git add .
git commit -m "Descriptive commit message"
```

## 📝 Code Guidelines

### Python Style

- **Formatter**: Black with 100 character line length
- **Import sorting**: isort with black profile
- **Type hints**: Required for public functions
- **Docstrings**: Google style for all classes and functions

### Example Code Style

```python
from typing import Dict, List, Optional, Tuple
import torch
import numpy as np

def process_pill_image(
    image_path: str,
    target_size: Tuple[int, int] = (224, 224),
    normalize: bool = True
) -> torch.Tensor:
    """
    Process a pill image for model input.
    
    Args:
        image_path: Path to the image file
        target_size: Target image dimensions (height, width)
        normalize: Whether to normalize pixel values
        
    Returns:
        Preprocessed image tensor
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image format is unsupported
    """
    # Implementation here...
    pass
```

### File Organization

```
feature/
├── __init__.py           # Module exports
├── main_module.py        # Main functionality
├── utils.py             # Helper functions
└── tests/
    ├── __init__.py
    ├── test_main_module.py
    └── test_utils.py
```

## 🧪 Testing

### Test Structure

- **Unit tests**: `tests/unit/`
- **Integration tests**: `tests/integration/`
- **E2E tests**: `tests/e2e/`

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_models.py -v

# With coverage
pytest tests/ -v --cov=core --cov-report=html

# Performance tests
pytest tests/test_performance.py -v --benchmark-only
```

### Writing Tests

```python
import pytest
import torch
from core.models.multimodal_transformer import MultimodalPillTransformer

class TestMultimodalTransformer:
    """Test cases for multimodal transformer."""
    
    @pytest.fixture
    def model(self):
        """Create a test model instance."""
        return MultimodalPillTransformer(
            num_classes=10,
            vision_model="vit_tiny_patch16_224"
        )
    
    def test_forward_pass(self, model):
        """Test model forward pass."""
        batch_size = 2
        image = torch.randn(batch_size, 3, 224, 224)
        text_ids = torch.randint(0, 1000, (batch_size, 128))
        
        output = model(image, text_ids)
        
        assert output.shape == (batch_size, 10)
        assert not torch.isnan(output).any()
```

## 📚 Documentation

### Types of Documentation

1. **Code Documentation**: Docstrings and type hints
2. **API Documentation**: Automatically generated from docstrings
3. **User Guides**: Markdown files in `docs/`
4. **README**: Main project documentation

### Documentation Standards

```python
class PillClassifier:
    """
    Multimodal pill classifier using vision and text.
    
    This class implements a transformer-based architecture that combines
    visual features from pill images with textual features from imprints
    for accurate pharmaceutical identification.
    
    Attributes:
        num_classes: Number of pill classes to classify
        vision_encoder: Vision transformer for image processing
        text_encoder: BERT encoder for text processing
        
    Example:
        >>> classifier = PillClassifier(num_classes=1000)
        >>> result = classifier.predict(image, text_imprint="ADVIL 200")
        >>> print(f"Prediction: {result['class_name']}")
    """
```

## 🔄 Pull Request Process

### Before Submitting

1. **Test**: Ensure all tests pass
2. **Format**: Code is properly formatted
3. **Document**: Update documentation if needed
4. **Changelog**: Add entry to CHANGELOG.md

### Pull Request Checklist

- [ ] Code follows style guidelines
- [ ] Tests added for new functionality
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Commit messages are descriptive
- [ ] No merge conflicts

### PR Template

```markdown
## 📝 Description
Brief description of changes made.

## 🔧 Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Refactoring

## 🧪 Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## 📚 Documentation
- [ ] Code documentation updated
- [ ] README updated if needed
- [ ] API docs updated

## 📋 Checklist
- [ ] Code follows project style
- [ ] Self-review completed
- [ ] Requested review from maintainers
```

## 🐛 Reporting Issues

### Bug Reports

Use the bug report template:

```markdown
**🐛 Bug Description**
Clear description of the bug.

**🔄 Steps to Reproduce**
1. Step one
2. Step two
3. Bug occurs

**✅ Expected Behavior**
What should happen.

**❌ Actual Behavior**
What actually happens.

**🖥️ Environment**
- OS: [e.g., Ubuntu 22.04]
- Python: [e.g., 3.10.12]
- PyTorch: [e.g., 2.3.0]
- CUDA: [e.g., 12.8]

**📎 Additional Context**
Screenshots, logs, or other context.
```

### Feature Requests

```markdown
**🚀 Feature Request**
Brief description of the feature.

**💡 Motivation**
Why is this feature needed?

**📋 Proposed Solution**
How should this feature work?

**🔄 Alternatives**
Alternative solutions considered.
```

## 🏷️ Versioning

We use [Semantic Versioning](https://semver.org/):

- `MAJOR.MINOR.PATCH`
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Recognition

Contributors are recognized in:
- README.md contributors section
- CHANGELOG.md for each release
- GitHub contributors page

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Documentation**: Check the full documentation first

Thank you for contributing to the Smart Pill Recognition System! 🎉