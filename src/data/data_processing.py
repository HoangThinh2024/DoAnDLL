import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, when, isnan, isnull
from pyspark.sql.types import StringType, BinaryType, ArrayType, FloatType
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, List, Tuple, Optional, Any
import io
import base64
from loguru import logger

try:
    import cudf
    import cupy as cp
    RAPIDS_AVAILABLE = True
    logger.info("Rapids CUDF/CuPy available for GPU acceleration")
except ImportError:
    RAPIDS_AVAILABLE = False
    logger.warning("Rapids not available, using CPU processing")


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
    
    def load_parquet_data(self, data_path: str) -> "pyspark.sql.DataFrame":
        """Load multimodal data from Parquet files"""
        try:
            df = self.spark.read.parquet(data_path)
            logger.info(f"Loaded {df.count()} records from {data_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading parquet data: {e}")
            raise
    
    def create_sample_dataset(self, output_path: str, num_samples: int = 1000):
        """Create a sample dataset for testing"""
        logger.info(f"Creating sample dataset with {num_samples} samples")
        
        # Generate synthetic data
        data = []
        pill_classes = [f"pill_class_{i}" for i in range(100)]
        
        for i in range(num_samples):
            # Generate dummy image data (base64 encoded)
            dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img_pil = Image.fromarray(dummy_img)
            
            # Convert to base64
            buffer = io.BytesIO()
            img_pil.save(buffer, format='JPEG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Generate text imprint
            text_samples = [
                f"PILL{i:04d}",
                f"MED{i:03d}",
                f"{i%10}{i%10}{i%10}",
                f"RX{i:02d}",
                f"TAB{i:03d}"
            ]
            
            data.append({
                "image_id": f"img_{i:06d}",
                "image_base64": img_base64,
                "text_imprint": text_samples[i % len(text_samples)],
                "pill_class": pill_classes[i % len(pill_classes)],
                "class_id": i % len(pill_classes)
            })
        
        # Create DataFrame
        df = self.spark.createDataFrame(data)
        
        # Save as Parquet
        df.write.mode("overwrite").parquet(output_path)
        logger.info(f"Sample dataset saved to {output_path}")
        
        return df
    
    def preprocess_images(self, df: "pyspark.sql.DataFrame") -> "pyspark.sql.DataFrame":
        """Preprocess images using Spark UDFs"""
        
        def decode_and_resize_image(image_base64: str) -> List[float]:
            """Decode base64 image and resize"""
            try:
                # Decode base64
                image_bytes = base64.b64decode(image_base64)
                image = Image.open(io.BytesIO(image_bytes))
                
                # Convert to RGB if necessary
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Resize
                image = image.resize((224, 224))
                
                # Convert to numpy and normalize
                image_array = np.array(image, dtype=np.float32) / 255.0
                
                # Flatten for storage in DataFrame
                return image_array.flatten().tolist()
                
            except Exception as e:
                logger.error(f"Error processing image: {e}")
                return [0.0] * (224 * 224 * 3)  # Return dummy data
        
        # Register UDF
        decode_udf = udf(decode_and_resize_image, ArrayType(FloatType()))
        
        # Apply preprocessing
        df_processed = df.withColumn("image_features", decode_udf(col("image_base64")))
        
        return df_processed
    
    def clean_text_data(self, df: "pyspark.sql.DataFrame") -> "pyspark.sql.DataFrame":
        """Clean and normalize text imprint data"""
        
        def clean_text(text: str) -> str:
            """Clean text imprint"""
            if text is None:
                return ""
            
            # Remove special characters, convert to uppercase
            cleaned = ''.join(c for c in text.upper() if c.isalnum() or c.isspace())
            cleaned = ' '.join(cleaned.split())  # Remove extra whitespace
            
            return cleaned
        
        clean_text_udf = udf(clean_text, StringType())
        
        df_cleaned = df.withColumn("text_imprint_clean", clean_text_udf(col("text_imprint")))
        
        # Filter out empty text
        df_cleaned = df_cleaned.filter(col("text_imprint_clean") != "")
        
        return df_cleaned
    
    def create_train_val_test_split(self, df: "pyspark.sql.DataFrame") -> Tuple:
        """Split data into train/validation/test sets"""
        train_ratio = self.config["data"]["train_split"]
        val_ratio = self.config["data"]["val_split"]
        test_ratio = self.config["data"]["test_split"]
        
        # Add random column for splitting
        df_with_rand = df.withColumn("rand", (col("class_id") % 10) / 10.0)
        
        # Split data
        train_df = df_with_rand.filter(col("rand") < train_ratio)
        val_df = df_with_rand.filter(
            (col("rand") >= train_ratio) & 
            (col("rand") < train_ratio + val_ratio)
        )
        test_df = df_with_rand.filter(col("rand") >= train_ratio + val_ratio)
        
        # Remove random column
        train_df = train_df.drop("rand")
        val_df = val_df.drop("rand")
        test_df = test_df.drop("rand")
        
        logger.info(f"Data split - Train: {train_df.count()}, "
                   f"Val: {val_df.count()}, Test: {test_df.count()}")
        
        return train_df, val_df, test_df
    
    def convert_to_pandas(self, df: "pyspark.sql.DataFrame") -> pd.DataFrame:
        """Convert Spark DataFrame to Pandas (for smaller datasets)"""
        if RAPIDS_AVAILABLE:
            # Use cuDF for GPU acceleration
            try:
                pandas_df = df.toPandas()
                cudf_df = cudf.from_pandas(pandas_df)
                logger.info("Converted to cuDF for GPU processing")
                return cudf_df
            except Exception as e:
                logger.warning(f"cuDF conversion failed: {e}, using pandas")
                return df.toPandas()
        else:
            return df.toPandas()
    
    def save_processed_data(self, train_df, val_df, test_df, output_dir: str):
        """Save processed data"""
        os.makedirs(output_dir, exist_ok=True)
        
        train_df.write.mode("overwrite").parquet(f"{output_dir}/train")
        val_df.write.mode("overwrite").parquet(f"{output_dir}/val")
        test_df.write.mode("overwrite").parquet(f"{output_dir}/test")
        
        logger.info(f"Processed data saved to {output_dir}")
    
    def close(self):
        """Close Spark session"""
        self.spark.stop()
        logger.info("Spark session closed")


class PillDataset(Dataset):
    """PyTorch Dataset for multimodal pill data"""
    
    def __init__(self, data_path: str, split: str = "train", 
                 config: Dict[str, Any] = None, transform=None):
        self.data_path = data_path
        self.split = split
        self.config = config or {}
        self.transform = transform
        
        # Load data
        self.data = self._load_data()
        
        # Create class to index mapping
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.data['pill_class'].unique())}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        logger.info(f"Loaded {len(self.data)} samples for {split} split")
    
    def _load_data(self) -> pd.DataFrame:
        """Load processed data"""
        parquet_path = os.path.join(self.data_path, self.split)
        
        if RAPIDS_AVAILABLE:
            try:
                return cudf.read_parquet(parquet_path)
            except:
                pass
        
        # Fallback to pandas
        import pyarrow.parquet as pq
        table = pq.read_table(parquet_path)
        return table.to_pandas()
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.data.iloc[idx]
        
        # Reconstruct image from features
        image_features = np.array(row['image_features'], dtype=np.float32)
        image = image_features.reshape(224, 224, 3)
        
        # Convert to PIL Image for transforms
        image_pil = Image.fromarray((image * 255).astype(np.uint8))
        
        # Apply transforms
        if self.transform:
            if isinstance(self.transform, A.Compose):
                # Albumentations
                image_array = np.array(image_pil)
                transformed = self.transform(image=image_array)
                image_tensor = transformed['image']
            else:
                # torchvision transforms
                image_tensor = self.transform(image_pil)
        else:
            # Default transform
            image_tensor = torch.from_numpy(image).permute(2, 0, 1)
        
        return {
            'image': image_tensor,
            'text': row['text_imprint_clean'],
            'label': self.class_to_idx[row['pill_class']],
            'class_name': row['pill_class'],
            'image_id': row['image_id']
        }


def get_data_transforms(config: Dict[str, Any]) -> Dict[str, A.Compose]:
    """Get data augmentation transforms"""
    
    # Training transforms with augmentation
    train_transform = A.Compose([
        A.Resize(config["data"]["image_size"], config["data"]["image_size"]),
        A.Rotate(limit=config["data"]["augmentation"]["rotation"], p=0.5),
        A.ColorJitter(
            brightness=config["data"]["augmentation"]["brightness"],
            contrast=config["data"]["augmentation"]["contrast"],
            saturation=config["data"]["augmentation"]["saturation"],
            hue=config["data"]["augmentation"]["hue"],
            p=0.5
        ),
        A.HorizontalFlip(p=config["data"]["augmentation"]["horizontal_flip"]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Validation/test transforms without augmentation
    val_transform = A.Compose([
        A.Resize(config["data"]["image_size"], config["data"]["image_size"]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    return {
        'train': train_transform,
        'val': val_transform,
        'test': val_transform
    }


def create_dataloaders(data_path: str, config: Dict[str, Any]) -> Dict[str, DataLoader]:
    """Create PyTorch DataLoaders"""
    
    transforms = get_data_transforms(config)
    
    # Create datasets
    datasets = {}
    for split in ['train', 'val', 'test']:
        datasets[split] = PillDataset(
            data_path=data_path,
            split=split,
            config=config,
            transform=transforms[split]
        )
    
    # Create dataloaders
    dataloaders = {}
    for split in ['train', 'val', 'test']:
        shuffle = (split == 'train')
        dataloaders[split] = DataLoader(
            datasets[split],
            batch_size=config["training"]["batch_size"],
            shuffle=shuffle,
            num_workers=4,
            pin_memory=torch.cuda.is_available(),
            collate_fn=lambda batch: custom_collate_fn(batch, datasets[split])
        )
    
    return dataloaders


def custom_collate_fn(batch: List[Dict], dataset: PillDataset) -> Dict[str, Any]:
    """Custom collate function for multimodal data"""
    
    images = torch.stack([item['image'] for item in batch])
    texts = [item['text'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    class_names = [item['class_name'] for item in batch]
    image_ids = [item['image_id'] for item in batch]
    
    return {
        'images': images,
        'texts': texts,
        'labels': labels,
        'class_names': class_names,
        'image_ids': image_ids
    }
