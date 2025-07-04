# 💊 Hệ thống Nhận dạng Viên Thuốc Multimodal với Transformer

## 📝 Tổng quan

Dự án này phát triển một hệ thống nhận dạng viên thuốc tiên tiến sử dụng **Multimodal Transformer** để kết hợp thông tin từ hình ảnh viên thuốc và text imprint (chữ in trên viên thuốc). Hệ thống áp dụng **Cross-modal Attention Mechanism** để học representation chung cho cả visual và textual features, đạt được độ chính xác cao trong việc phân loại và nhận dạng viên thuốc.

## 🎯 Mục tiêu

- Phát triển hệ thống multimodal fusion cho nhận dạng viên thuốc từ hình ảnh và text
- Áp dụng kiến trúc CLIP-like với Cross-modal attention mechanism
- Xử lý dữ liệu lớn với Apache Spark và GPU acceleration
- Tạo giao diện người dùng trực quan với Streamlit

## 🏗️ Kiến trúc Hệ thống

### 1. Multimodal Transformer Architecture

```
┌─────────────────┐    ┌─────────────────┐
│   Image Input   │    │   Text Input    │
│   (224x224x3)   │    │   (Imprint)     │
└─────────┬───────┘    └─────────┬───────┘
          │                      │
          ▼                      ▼
┌─────────────────┐    ┌─────────────────┐
│ Visual Encoder  │    │  Text Encoder   │
│     (ViT)       │    │     (BERT)      │
└─────────┬───────┘    └─────────┬───────┘
          │                      │
          ▼                      ▼
┌─────────────────┐    ┌─────────────────┐
│ Visual Features │    │ Text Features   │
│   (768 dims)    │    │   (768 dims)    │
└─────────┬───────┘    └─────────┬───────┘
          │                      │
          └──────────┬───────────┘
                     ▼
           ┌─────────────────┐
           │ Cross-modal     │
           │ Attention       │
           │ Fusion          │
           └─────────┬───────┘
                     ▼
           ┌─────────────────┐
           │   Classifier    │
           │ (Pill Classes)  │
           └─────────────────┘
```

### 2. Các thành phần chính

- **Visual Encoder**: Vision Transformer (ViT) hoặc CNN backbone
- **Text Encoder**: BERT-based transformer cho text imprint
- **Cross-modal Attention**: Mechanism kết hợp thông tin từ hai modality
- **Fusion Layer**: Tổng hợp features từ visual và text
- **Classifier**: Phân loại viên thuốc cuối cùng

## 🚀 Tech Stack

### Core ML/DL Frameworks
- **PyTorch** 2.0+: Deep learning framework chính
- **Transformers** 4.30+: BERT và ViT models
- **timm**: Vision models pretrained
- **torchvision**: Computer vision utilities

### Big Data & Distributed Computing
- **Apache Spark** 3.4+: Xử lý dữ liệu phân tán
- **Rapids cuDF/cuML**: GPU acceleration cho pandas operations
- **Apache Parquet**: Columnar storage format
- **Elasticsearch**: Text indexing và search

### UI & Visualization
- **Streamlit** 1.25+: Web application framework
- **Plotly**: Interactive charts và graphs
- **streamlit-option-menu**: Enhanced navigation

### Data Processing
- **Pandas** 2.0+: Data manipulation
- **NumPy**: Numerical computing
- **Pillow**: Image processing
- **OpenCV**: Computer vision
- **Albumentations**: Data augmentation

## 📦 Cài đặt

### 1. Clone repository

```bash
git clone https://github.com/username/DoAnDLL.git
cd DoAnDLL
```

### 2. Tạo virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate     # Windows
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 4. Cài đặt Rapids (tùy chọn, cho GPU acceleration)

```bash
# Chỉ dành cho hệ thống có GPU NVIDIA
conda install -c rapidsai -c nvidia -c conda-forge cudf cuml
```

### 5. Setup Spark (tùy chọn)

```bash
# Download và setup Apache Spark
wget https://downloads.apache.org/spark/spark-3.4.0/spark-3.4.0-bin-hadoop3.tgz
tar -xzf spark-3.4.0-bin-hadoop3.tgz
export SPARK_HOME=/path/to/spark-3.4.0-bin-hadoop3
export PATH=$PATH:$SPARK_HOME/bin
```

## 🎮 Sử dụng

### 1. Chạy ứng dụng Streamlit

```bash
streamlit run app.py
```

Ứng dụng sẽ chạy trên `http://localhost:8501`

### 2. Training model

```bash
python src/training/trainer.py
```

### 3. Xử lý dữ liệu với Spark

```python
from src.data.data_processing import SparkDataProcessor
import yaml

# Load config
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize processor
processor = SparkDataProcessor(config)

# Create sample dataset
processor.create_sample_dataset("data/raw/sample.parquet", 1000)

# Process data
df = processor.load_parquet_data("data/raw/sample.parquet")
processed_df = processor.preprocess_images(df)
processed_df = processor.clean_text_data(processed_df)

# Split and save
train_df, val_df, test_df = processor.create_train_val_test_split(processed_df)
processor.save_processed_data(train_df, val_df, test_df, "data/processed")
```

## 📊 Giao diện Streamlit

Ứng dụng Streamlit bao gồm các trang:

### 🏠 Trang chủ
- Giới thiệu hệ thống
- Thống kê tổng quan
- Hướng dẫn sử dụng

### 🔍 Nhận dạng
- Upload hình ảnh viên thuốc
- Nhập text imprint
- Hiển thị kết quả dự đoán với độ tin cậy
- Phân tích features multimodal

### 📊 Thống kê
- Biểu đồ phân bố dữ liệu
- Metrics hiệu suất model
- Quá trình training

### ℹ️ Thông tin
- Kiến trúc hệ thống
- Cấu hình model
- Thông tin nhóm phát triển

## 🗂️ Cấu trúc Dự án

```
DoAnDLL/
├── app.py                      # Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # Documentation
├── LICENSE                     # License file
│
├── config/
│   └── config.yaml            # Configuration file
│
├── src/
│   ├── models/
│   │   └── multimodal_transformer.py  # Model architecture
│   │
│   ├── data/
│   │   └── data_processing.py         # Data processing với Spark
│   │
│   ├── training/
│   │   └── trainer.py                 # Training pipeline
│   │
│   └── utils/
│       ├── utils.py                   # Utility functions
│       └── metrics.py                 # Evaluation metrics
│
├── data/
│   ├── raw/                   # Raw data
│   └── processed/             # Processed data
│
├── checkpoints/               # Model checkpoints
├── logs/                     # Training logs
└── notebooks/                # Jupyter notebooks
```

## ⚙️ Cấu hình

File `config/config.yaml` chứa tất cả cấu hình:

```yaml
model:
  visual_encoder:
    type: "vit"
    model_name: "vit_base_patch16_224"
  text_encoder:
    type: "bert"
    model_name: "bert-base-uncased"
  fusion:
    type: "cross_attention"
    num_attention_heads: 8

training:
  batch_size: 32
  learning_rate: 1e-4
  num_epochs: 100

data:
  image_size: 224
  spark:
    app_name: "PillRecognitionETL"
    master: "local[*]"
```

## 🧪 Dataset

Hệ thống hỗ trợ xử lý dataset viên thuốc với:

- **Hình ảnh**: Format JPG/PNG, resize về 224x224
- **Text imprint**: Text được in trên viên thuốc
- **Labels**: Phân loại viên thuốc
- **Metadata**: Thông tin bổ sung (liều lượng, nhà sản xuất, etc.)

### Định dạng dữ liệu

```json
{
  "image_id": "img_000001",
  "image_path": "path/to/image.jpg",
  "text_imprint": "PILL123",
  "pill_class": "Acetaminophen 500mg",
  "class_id": 0,
  "metadata": {
    "dosage": "500mg",
    "manufacturer": "Company A"
  }
}
```

## 🏋️ Training

### 1. Chuẩn bị dữ liệu

```bash
python src/data/data_processing.py
```

### 2. Training model

```bash
python src/training/trainer.py --config config/config.yaml
```

### 3. Theo dõi training với Weights & Biases

```bash
# Setup wandb
wandb login
wandb init
```

## 📈 Evaluation Metrics

Hệ thống đánh giá model bằng các metrics:

- **Accuracy**: Độ chính xác tổng thể
- **Precision/Recall/F1**: Cho từng class
- **Top-k Accuracy**: Accuracy trong top-k predictions
- **Confusion Matrix**: Ma trận nhầm lẫn
- **Cross-modal Similarity**: Độ tương đồng giữa visual và text features

## 🚀 Deployment

### Docker

```dockerfile
FROM python:3.8-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
```

### Build và chạy

```bash
docker build -t pill-recognition .
docker run -p 8501:8501 pill-recognition
```

## 🤝 Đóng góp

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push branch (`git push origin feature/AmazingFeature`)
5. Tạo Pull Request

## 📄 License

Dự án được phân phối dưới MIT License. Xem `LICENSE` để biết thêm thông tin.

## 🙏 Acknowledgments

- [Transformers](https://huggingface.co/transformers/) - BERT và ViT models
- [timm](https://github.com/rwightman/pytorch-image-models) - Vision models
- [Apache Spark](https://spark.apache.org/) - Big data processing
- [Rapids](https://rapids.ai/) - GPU acceleration
- [Streamlit](https://streamlit.io/) - Web application framework

## 📞 Liên hệ

- **Tác giả**: [Tên sinh viên]
- **Email**: [email@example.com]
- **GitHub**: [https://github.com/username](https://github.com/username)
- **LinkedIn**: [https://linkedin.com/in/username](https://linkedin.com/in/username)

---

⭐ Nếu dự án này hữu ích, hãy cho chúng tôi một star!