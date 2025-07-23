# Smart Pill Recognition System - Core Package

Core AI modules for pharmaceutical identification using multimodal transformers.

## Modules

- **models/**: Neural network architectures
- **data/**: Dataset handling and preprocessing  
- **training/**: Training procedures and optimizers
- **utils/**: Utility functions and helpers

## Usage

```python
from core.models.multimodal_transformer import MultimodalPillTransformer
from core.data.data_processing import preprocess_image

# Load model
model = MultimodalPillTransformer.load_pretrained()

# Process image
image_tensor = preprocess_image(image_path)

# Make prediction
result = model.predict(image_tensor, text_imprint="ADVIL 200")
```
