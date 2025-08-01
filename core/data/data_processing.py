import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, List, Tuple, Optional, Any
import io
import base64
import json
from pathlib import Path
from loguru import logger

try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, udf, when, isnan, isnull
    from pyspark.sql.types import StringType, BinaryType, ArrayType, FloatType
    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False
    logger.warning("PySpark not available")

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import cudf
    import cupy as cp
    RAPIDS_AVAILABLE = True
    logger.info("Rapids CUDF/CuPy available for GPU acceleration")
except ImportError:
    RAPIDS_AVAILABLE = False
    logger.warning("Rapids not available, using CPU processing")


class UnifiedPillDataset(Dataset):
    """
    Unified dataset class that works with all training methods
    """
    
    def __init__(self,
                 data_path: str,
                 transform: Optional[A.Compose] = None,
                 tokenizer: Optional[Any] = None,
                 max_text_length: int = 128,
                 image_size: int = 224,
                 split: str = 'train'):
        
        self.data_path = Path(data_path)
        self.transform = transform or self._get_default_transform(image_size)
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.split = split
        
        # Load data
        self.data = self._load_data()
        if len(self.data) == 0:
            raise ValueError(f"❌ Không tìm thấy dữ liệu trong '{self.data_path}'. Bạn phải truyền đường dẫn dữ liệu thật, không được để trống!")
        print(f"✅ Dataset loaded: {len(self.data)} samples ({split})")
    
    def _get_default_transform(self, image_size: int) -> A.Compose:
        """Get default image transforms"""
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data from various formats"""
        if self.data_path.is_file():
            if self.data_path.suffix.lower() == '.json':
                return self._load_from_json()
            elif self.data_path.suffix.lower() == '.csv':
                return self._load_from_csv()
            elif self.data_path.suffix.lower() == '.parquet':
                return self._load_from_parquet()
            else:
                raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
        elif self.data_path.is_dir():
            return self._load_from_directory()
        else:
            raise ValueError(f"Invalid data path: {self.data_path}")
    
    def _load_from_json(self) -> List[Dict[str, Any]]:
        """Load data from JSON file"""
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # Assume it's split by train/val/test
            return data.get(self.split, [])
        else:
            raise ValueError("Invalid JSON format")
    
    def _load_from_csv(self) -> List[Dict[str, Any]]:
        """Load data from CSV file"""
        df = pd.read_csv(self.data_path)
        return df.to_dict('records')
    
    def _load_from_parquet(self) -> List[Dict[str, Any]]:
        """Load data from Parquet file"""
        df = pd.read_parquet(self.data_path)
        return df.to_dict('records')
    
    def _load_from_directory(self) -> List[Dict[str, Any]]:
        """Load data from directory structure"""
        data = []
        
        # Assume directory structure: data_path/class_name/images
        for class_dir in self.data_path.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                class_id = hash(class_name) % 1000  # Simple class ID generation
                
                for img_file in class_dir.glob('*.jpg'):
                    data.append({
                        'image_path': str(img_file),
                        'text_imprint': class_name,  # Use class name as text
                        'label': class_id,
                        'class_name': class_name
                    })
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        
        # Load and process image
        image_path = item.get('image_path', '')
        if image_path and Path(image_path).exists():
            image = Image.open(image_path).convert('RGB')
        elif 'image_data' in item:
            # Handle base64 encoded images
            image = self._decode_base64_image(item['image_data'])
        else:
            # Create dummy image if path doesn't exist
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        # Convert PIL to numpy for albumentations
        image = np.array(image)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            # Basic transform if no transform provided
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # Process text
        text = item.get('text_imprint', item.get('text', item.get('cleaned_text', '')))
        
        result = {
            'image': image,
            'text': text,
            'label': item.get('label', 0),
            'image_path': image_path
        }
        
        # Add tokenized text if tokenizer provided
        if self.tokenizer:
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_text_length,
                return_tensors='pt'
            )
            result.update({
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze()
            })
        
        return result
    
    def _decode_base64_image(self, image_b64: str) -> Image.Image:
        """Decode base64 image string"""
        try:
            image_bytes = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            return image
        except Exception as e:
            logger.error(f"Error decoding base64 image: {e}")
            return Image.new('RGB', (224, 224), color=(128, 128, 128))


class DataProcessor:
    """
    Unified data processor for all training methods
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_config = config.get('data', {})
        
    def prepare_pytorch_data(self, 
                           data_path: str,
                           batch_size: int = 32,
                           num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Prepare data for PyTorch training
        """
        # Get tokenizer if available
        tokenizer = None
        if TRANSFORMERS_AVAILABLE:
            text_model = self.config.get('model', {}).get('text_encoder', {}).get('model_name', 'bert-base-uncased')
            try:
                tokenizer = AutoTokenizer.from_pretrained(text_model)
            except:
                tokenizer = None
        
        # Get transforms
        train_transform = get_data_transforms(self.config, "train")
        val_transform = get_data_transforms(self.config, "val")
        
        # Create datasets
        image_size = self.data_config.get('image_size', 224)
        max_text_length = self.config.get('model', {}).get('text_encoder', {}).get('max_length', 128)
        
        # For simplicity, we'll use the same dataset for all splits
        full_dataset = UnifiedPillDataset(
            data_path=data_path,
            transform=train_transform,
            tokenizer=tokenizer,
            max_text_length=max_text_length,
            image_size=image_size
        )
        
        # Split dataset
        train_size = int(0.8 * len(full_dataset))
        val_size = int(0.1 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size, test_size]
        )
        
        # Update transforms for validation/test
        val_dataset.dataset.transform = val_transform
        test_dataset.dataset.transform = val_transform
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        print(f"✅ PyTorch data prepared: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
        
        return train_loader, val_loader, test_loader
    
    def prepare_spark_data(self, data_path: str) -> Optional[Any]:
        """Prepare data for Spark training"""
        if not PYSPARK_AVAILABLE:
            print("❌ PySpark not available")
            return None
        
        # Use existing SparkDataProcessor for complex operations
        spark_processor = SparkDataProcessor(self.config)
        return spark_processor.load_parquet_data(data_path)
    
    def prepare_hf_data(self, data_path: str) -> List[Dict[str, Any]]:
        """Prepare data for HuggingFace Transformers training"""
        dataset = UnifiedPillDataset(data_path)
        
        # Convert to format expected by HuggingFace
        hf_data = []
        for item in dataset.data:
            hf_data.append({
                'image_path': item.get('image_path', ''),
                'text': item.get('text_imprint', item.get('text', '')),
                'label': item.get('label', 0)
            })
        
        print(f"✅ HuggingFace data prepared: {len(hf_data)} samples")
        return hf_data
    
    def create_dummy_dataset(self, 
                           output_path: str,
                           num_samples: int = 1000,
                           num_classes: int = 10) -> str:
        """Create dummy dataset for testing"""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create dummy data
        dummy_data = []
        class_names = [f"pill_class_{i}" for i in range(num_classes)]
        
        for i in range(num_samples):
            class_id = i % num_classes
            class_name = class_names[class_id]
            
            dummy_data.append({
                'image_path': f'/dummy/path/image_{i}.jpg',
                'text_imprint': f'PILL{i:03d}',
                'text': f'Pill {class_name} imprint PILL{i:03d}',
                'label': class_id,
                'class_name': class_name
            })
        
        # Save as JSON
        json_path = output_path / 'dataset.json'
        with open(json_path, 'w') as f:
            json.dump(dummy_data, f, indent=2)
        
        # Save as CSV
        csv_path = output_path / 'dataset.csv'
        pd.DataFrame(dummy_data).to_csv(csv_path, index=False)
        
        print(f"✅ Dummy dataset created: {json_path}")
        print(f"📊 Samples: {num_samples}, Classes: {num_classes}")
        
        return str(json_path)


class SparkDataProcessor:
    """Spark-based data processor for large-scale multimodal data"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.spark_config = config["data"]["spark"]
        
        # Initialize Spark session
        self.spark = SparkSession.builder \
            .appName(self.spark_config["app_name"]) \
            .master(self.spark_config["master"]) \
            .config("spark.executor.memory", self.spark_config["executor_memory"]) \
            .config("spark.driver.memory", self.spark_config["driver_memory"]) \
            .config("spark.driver.maxResultSize", self.spark_config["max_result_size"]) \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .getOrCreate()
            
        if RAPIDS_AVAILABLE:
            # Configure Spark for GPU acceleration
            self.spark.conf.set("spark.rapids.sql.enabled", "true")
            self.spark.conf.set("spark.plugins", "com.nvidia.spark.SQLPlugin")
            
        logger.info(f"Spark session initialized: {self.spark.sparkContext.appName}")
    
    def create_sample_dataset(self, output_path: str, num_samples: int = 10000):
        """Create a sample synthetic dataset for testing"""
        logger.info(f"Creating sample dataset with {num_samples} samples")
        
        # Generate synthetic data
        data = []
        pill_classes = [f"pill_class_{i:04d}" for i in range(100)]
        
        for i in range(num_samples):
            # Generate random pill data
            pill_class = np.random.choice(pill_classes)
            
            # Synthetic image (would be replaced with real image paths in production)
            image_data = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            image_bytes = cv2.imencode('.jpg', image_data)[1].tobytes()
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # Generate synthetic text imprint
            text_patterns = ["PILL", "MED", "TAB", "CAP", "RX"]
            numbers = [str(np.random.randint(1, 999)) for _ in range(2)]
            text_imprint = f"{np.random.choice(text_patterns)} {' '.join(numbers)}"
            
            data.append({
                "id": f"pill_{i:06d}",
                "image_data": image_b64,
                "text_imprint": text_imprint,
                "pill_class": pill_class,
                "manufacturer": f"pharma_company_{np.random.randint(1, 20)}",
                "dosage": f"{np.random.randint(5, 500)}mg",
                "shape": np.random.choice(["round", "oval", "square", "capsule"]),
                "color": np.random.choice(["white", "blue", "red", "yellow", "green"]),
                "split": np.random.choice(["train", "val", "test"], p=[0.7, 0.15, 0.15])
            })
        
        # Convert to DataFrame and save
        df = pd.DataFrame(data)
        
        if RAPIDS_AVAILABLE:
            # Use cuDF for GPU acceleration
            cudf_df = cudf.from_pandas(df)
            spark_df = self.spark.createDataFrame(df)
        else:
            spark_df = self.spark.createDataFrame(df)
        
        # Save as Parquet for efficient storage
        spark_df.write.mode("overwrite").parquet(output_path)
        logger.info(f"Sample dataset saved to {output_path}")
        
        return spark_df
    
    def load_parquet_data(self, data_path: str):
        """Load data from Parquet format"""
        logger.info(f"Loading data from {data_path}")
        df = self.spark.read.parquet(data_path)
        logger.info(f"Loaded {df.count()} samples")
        return df
    
    def preprocess_images(self, df):
        """Preprocess image data using Spark UDFs"""
        
        def decode_and_preprocess_image(image_b64_str):
            """Decode base64 image and apply preprocessing"""
            try:
                # Decode base64
                image_bytes = base64.b64decode(image_b64_str)
                image_array = np.frombuffer(image_bytes, dtype=np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                
                # Resize to target size
                image = cv2.resize(image, (224, 224))
                
                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Normalize to [0, 1]
                image = image.astype(np.float32) / 255.0
                
                # Convert back to bytes for storage
                image_bytes = image.tobytes()
                return base64.b64encode(image_bytes).decode('utf-8')
            except Exception as e:
                logger.error(f"Error processing image: {e}")
                return None
        
        # Register UDF
        preprocess_udf = udf(decode_and_preprocess_image, StringType())
        
        # Apply preprocessing
        df = df.withColumn("processed_image", preprocess_udf(col("image_data")))
        
        # Filter out failed preprocessings
        df = df.filter(col("processed_image").isNotNull())
        
        logger.info("Image preprocessing completed")
        return df
    
    def clean_text_data(self, df):
        """Clean and preprocess text imprint data"""
        
        def clean_text(text):
            """Clean text imprint"""
            if text is None:
                return ""
            
            # Convert to uppercase
            text = text.upper()
            
            # Remove special characters but keep alphanumeric and spaces
            import re
            text = re.sub(r'[^A-Z0-9\s]', ' ', text)
            
            # Remove extra whitespace
            text = ' '.join(text.split())
            
            return text
        
        clean_text_udf = udf(clean_text, StringType())
        
        # Apply text cleaning
        df = df.withColumn("cleaned_text", clean_text_udf(col("text_imprint")))
        
        # Filter out empty text
        df = df.filter(col("cleaned_text") != "")
        
        logger.info("Text cleaning completed")
        return df
    
    def create_train_val_test_split(self, df, train_ratio=0.7, val_ratio=0.15):
        """Create train/validation/test splits"""
        
        # If split column exists, use it
        if "split" in df.columns:
            train_df = df.filter(col("split") == "train")
            val_df = df.filter(col("split") == "val") 
            test_df = df.filter(col("split") == "test")
        else:
            # Create random splits
            test_ratio = 1.0 - train_ratio - val_ratio
            train_df, val_df, test_df = df.randomSplit([train_ratio, val_ratio, test_ratio], seed=42)
        
        train_count = train_df.count()
        val_count = val_df.count()
        test_count = test_df.count()
        
        logger.info(f"Dataset split - Train: {train_count}, Val: {val_count}, Test: {test_count}")
        
        return train_df, val_df, test_df
    
    def save_processed_data(self, train_df, val_df, test_df, output_dir: str):
        """Save processed data splits"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save each split
        train_df.write.mode("overwrite").parquet(f"{output_dir}/train")
        val_df.write.mode("overwrite").parquet(f"{output_dir}/val")
        test_df.write.mode("overwrite").parquet(f"{output_dir}/test")
        
        logger.info(f"Processed data saved to {output_dir}")
    
    def get_data_statistics(self, df):
        """Get comprehensive data statistics"""
        stats = {}
        
        # Basic stats
        stats['total_samples'] = df.count()
        stats['unique_classes'] = df.select("pill_class").distinct().count()
        
        # Class distribution
        class_dist = df.groupBy("pill_class").count().orderBy(col("count").desc())
        stats['class_distribution'] = class_dist.collect()
        
        # Text length statistics
        text_lengths = df.select(col("cleaned_text")).rdd.map(lambda row: len(row[0].split()) if row[0] else 0)
        stats['avg_text_length'] = text_lengths.mean()
        stats['max_text_length'] = text_lengths.max()
        stats['min_text_length'] = text_lengths.min()
        
        return stats


class PillDataset(Dataset):
    """PyTorch dataset for pill recognition"""
    
    def __init__(self, 
                 data_path: str,
                 transform=None,
                 tokenizer=None,
                 max_length: int = 128,
                 mode: str = "train"):
        
        self.data_path = data_path
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        
        # Load data
        self.data = self._load_data()
        self.class_to_idx = self._create_class_mapping()
    
    def _load_data(self):
        """Load data from parquet files"""
        if os.path.isdir(self.data_path):
            # Load from directory (train/val/test splits)
            data_files = []
            for file in os.listdir(self.data_path):
                if file.endswith('.parquet'):
                    data_files.append(os.path.join(self.data_path, file))
            
            if data_files:
                df = pd.read_parquet(data_files[0])
            else:
                raise ValueError(f"No parquet files found in {self.data_path}")
        else:
            # Load single file
            df = pd.read_parquet(self.data_path)
        
        return df
    
    def _create_class_mapping(self):
        """Create class to index mapping"""
        unique_classes = sorted(self.data['pill_class'].unique())
        return {cls: idx for idx, cls in enumerate(unique_classes)}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load and process image
        image_b64 = row['processed_image'] if 'processed_image' in row else row['image_data']
        image = self._decode_image(image_b64)
        
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform
            image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        # Process text
        text = row['cleaned_text'] if 'cleaned_text' in row else row['text_imprint']
        if self.tokenizer:
            text_encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = text_encoding['input_ids'].squeeze()
            attention_mask = text_encoding['attention_mask'].squeeze()
        else:
            # Simple tokenization
            input_ids = torch.zeros(self.max_length, dtype=torch.long)
            attention_mask = torch.ones(self.max_length, dtype=torch.long)
        
        # Get label
        label = self.class_to_idx[row['pill_class']]
        
        return {
            'image': image,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.long),
            'pill_class': row['pill_class'],
            'text_imprint': text
        }
    
    def _decode_image(self, image_b64):
        """Decode base64 image"""
        try:
            image_bytes = base64.b64decode(image_b64)
            
            # Check if this is preprocessed data (numpy array) or raw image
            if len(image_bytes) == 224 * 224 * 3 * 4:  # float32 array
                image = np.frombuffer(image_bytes, dtype=np.float32).reshape(224, 224, 3)
            else:
                # Raw image bytes
                image_array = np.frombuffer(image_bytes, dtype=np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                image = cv2.resize(image, (224, 224))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = image.astype(np.float32) / 255.0
            
            return image
        except Exception as e:
            logger.error(f"Error decoding image: {e}")
            # Return dummy image
            return np.zeros((224, 224, 3), dtype=np.float32)


def get_data_transforms(config: Dict[str, Any], mode: str = "train"):
    """Get data transforms for training/validation"""
    
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        
        aug_config = config["data"]["augmentation"]
        
        if mode == "train":
            transform = A.Compose([
                A.HorizontalFlip(p=aug_config.get("horizontal_flip", 0.5)),
                A.Rotate(limit=aug_config.get("rotation", 15), p=0.7),
                A.ColorJitter(
                    brightness=aug_config.get("color_jitter", {}).get("brightness", 0.2),
                    contrast=aug_config.get("color_jitter", {}).get("contrast", 0.2),
                    saturation=aug_config.get("color_jitter", {}).get("saturation", 0.2),
                    hue=aug_config.get("color_jitter", {}).get("hue", 0.1),
                    p=0.5
                ),
                A.GaussNoise(var_limit=(10, 50), p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            transform = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        
        return transform
        
    except ImportError:
        logger.warning("Albumentations not available, using basic transforms")
        return None


def create_dataloaders(config: Dict[str, Any], tokenizer=None):
    """Create train/val/test dataloaders"""
    
    data_config = config["data"]
    training_config = config["training"]
    
    # Get transforms
    train_transform = get_data_transforms(config, "train")
    val_transform = get_data_transforms(config, "val")
    
    # Create datasets
    train_dataset = PillDataset(
        data_path=os.path.join(data_config["data_path"], "processed/train"),
        transform=train_transform,
        tokenizer=tokenizer,
        max_length=config["model"]["text_encoder"]["max_length"],
        mode="train"
    )
    
    val_dataset = PillDataset(
        data_path=os.path.join(data_config["data_path"], "processed/val"),
        transform=val_transform,
        tokenizer=tokenizer,
        max_length=config["model"]["text_encoder"]["max_length"],
        mode="val"
    )
    
    test_dataset = PillDataset(
        data_path=os.path.join(data_config["data_path"], "processed/test"),
        transform=val_transform,
        tokenizer=tokenizer,
        max_length=config["model"]["text_encoder"]["max_length"],
        mode="test"
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
        num_workers=training_config.get("num_workers", 4),
        pin_memory=training_config.get("pin_memory", True),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.get("val_batch_size", training_config["batch_size"]),
        shuffle=False,
        num_workers=training_config.get("num_workers", 4),
        pin_memory=training_config.get("pin_memory", True)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=training_config.get("val_batch_size", training_config["batch_size"]),
        shuffle=False,
        num_workers=training_config.get("num_workers", 4),
        pin_memory=training_config.get("pin_memory", True)
    )
    
    return train_loader, val_loader, test_loader, train_dataset.class_to_idx
