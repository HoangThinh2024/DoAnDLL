"""
PySpark Distributed Training Module for Smart Pill Recognition

This module implements distributed training using Apache Spark for processing
large datasets and training deep learning models at scale.

Author: DoAnDLL Team  
Date: 2024
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import json
import pickle
from datetime import datetime

# PySpark imports
try:
    import findspark
    findspark.init()
    
    from pyspark.sql import SparkSession, DataFrame
    from pyspark.sql.functions import col, when, isnan, count, lit, udf
    from pyspark.sql.types import *
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
    from pyspark.ml.classification import MultilayerPerceptronClassifier, RandomForestClassifier
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
    from pyspark.ml.image import ImageSchema
    
    PYSPARK_AVAILABLE = True
except ImportError as e:
    print(f"PySpark not available: {e}")
    PYSPARK_AVAILABLE = False

# Deep learning libraries for distributed training
try:
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

# Additional ML libraries
try:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ..models.model_registry import ModelRegistry, TrainingMethod
from ..utils.metrics import MetricsCalculator


class SparkImageProcessor:
    """Process images using Spark for distributed computing"""
    
    def __init__(self, spark_session: SparkSession):
        self.spark = spark_session
    
    def load_images_from_directory(self, 
                                  image_dir: str, 
                                  label_mapping: Dict[str, int] = None) -> DataFrame:
        """
        Load images from directory structure using Spark
        
        Args:
            image_dir: Root directory containing images
            label_mapping: Mapping from folder names to class indices
            
        Returns:
            Spark DataFrame with images and labels
        """
        try:
            # Load images using Spark's ImageSchema
            df = ImageSchema.readImages(image_dir, recursive=True)
            
            # Extract labels from path
            extract_label_udf = udf(lambda path: self._extract_label_from_path(path, label_mapping), StringType())
            df = df.withColumn("label_str", extract_label_udf(col("image.origin")))
            
            # Convert string labels to indices
            if label_mapping:
                mapping_expr = create_map([lit(x) for x in chain(*label_mapping.items())])
                df = df.withColumn("label", mapping_expr[col("label_str")])
            else:
                indexer = StringIndexer(inputCol="label_str", outputCol="label")
                df = indexer.fit(df).transform(df)
            
            return df
            
        except Exception as e:
            print(f"Error loading images: {e}")
            return None
    
    def _extract_label_from_path(self, path: str, label_mapping: Dict[str, int] = None) -> str:
        """Extract label from image path"""
        # Assuming directory structure: /path/to/dataset/class_name/image.jpg
        parts = path.split('/')
        if len(parts) >= 2:
            return parts[-2]  # Parent directory name
        return "unknown"
    
    def extract_image_features(self, df: DataFrame, feature_extractor_path: str = None) -> DataFrame:
        """
        Extract features from images using pre-trained model
        
        Args:
            df: DataFrame with images
            feature_extractor_path: Path to feature extractor model
            
        Returns:
            DataFrame with extracted features
        """
        # This would typically use a distributed feature extraction approach
        # For now, we'll implement a simplified version
        
        def extract_features_udf(image_data):
            """UDF for feature extraction"""
            try:
                # Convert image to numpy array
                height = image_data["height"]
                width = image_data["width"]
                channels = image_data["nChannels"]
                data = image_data["data"]
                
                # Simple feature extraction (mean, std, etc.)
                # In production, this would use a CNN feature extractor
                img_array = np.frombuffer(data, dtype=np.uint8)
                img_array = img_array.reshape((height, width, channels))
                
                # Extract basic statistical features
                features = [
                    float(np.mean(img_array)),
                    float(np.std(img_array)),
                    float(np.min(img_array)),
                    float(np.max(img_array)),
                    float(np.median(img_array))
                ]
                
                # Add color channel statistics
                for c in range(channels):
                    channel_data = img_array[:, :, c]
                    features.extend([
                        float(np.mean(channel_data)),
                        float(np.std(channel_data))
                    ])
                
                return features
                
            except Exception as e:
                print(f"Feature extraction error: {e}")
                return [0.0] * 11  # Return default features
        
        # Register UDF
        extract_features_spark_udf = udf(extract_features_udf, ArrayType(DoubleType()))
        
        # Apply feature extraction
        df_with_features = df.withColumn("features", extract_features_spark_udf(col("image")))
        
        return df_with_features


class SparkTextProcessor:
    """Process text data using Spark"""
    
    def __init__(self, spark_session: SparkSession):
        self.spark = spark_session
    
    def process_text_imprints(self, df: DataFrame, text_column: str = "text_imprint") -> DataFrame:
        """
        Process text imprints using Spark
        
        Args:
            df: DataFrame with text data
            text_column: Name of text column
            
        Returns:
            DataFrame with processed text features
        """
        try:
            from pyspark.ml.feature import Tokenizer, HashingTF, IDF
            
            # Tokenization
            tokenizer = Tokenizer(inputCol=text_column, outputCol="words")
            df_tokenized = tokenizer.transform(df)
            
            # TF-IDF
            hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=1000)
            df_tf = hashingTF.transform(df_tokenized)
            
            idf = IDF(inputCol="rawFeatures", outputCol="text_features")
            idf_model = idf.fit(df_tf)
            df_tfidf = idf_model.transform(df_tf)
            
            return df_tfidf
            
        except Exception as e:
            print(f"Text processing error: {e}")
            return df


class SparkMultimodalTrainer:
    """
    Distributed multimodal training using Apache Spark
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.spark = self._initialize_spark()
        self.image_processor = SparkImageProcessor(self.spark)
        self.text_processor = SparkTextProcessor(self.spark)
        self.metrics_calculator = MetricsCalculator(
            num_classes=config.get("model", {}).get("classifier", {}).get("num_classes", 1000)
        )
        
        # Model registry
        self.model_registry = ModelRegistry()
        
        print("✅ Spark Multimodal Trainer initialized")
    
    def _initialize_spark(self) -> SparkSession:
        """Initialize Spark session with optimized configuration"""
        spark_config = self.config.get("data", {}).get("spark", {})
        
        builder = SparkSession.builder
        builder = builder.appName(spark_config.get("app_name", "PillRecognitionSpark"))
        builder = builder.master(spark_config.get("master", "local[*]"))
        
        # Memory configuration
        builder = builder.config("spark.executor.memory", spark_config.get("executor_memory", "8g"))
        builder = builder.config("spark.driver.memory", spark_config.get("driver_memory", "4g"))
        builder = builder.config("spark.sql.execution.arrow.pyspark.enabled", "true")
        
        # Performance optimization
        builder = builder.config("spark.sql.adaptive.enabled", "true")
        builder = builder.config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        builder = builder.config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        
        spark = builder.getOrCreate()
        
        # Set log level
        spark.sparkContext.setLogLevel("WARN")
        
        print(f"✅ Spark session initialized with {spark.sparkContext.defaultParallelism} cores")
        return spark
    
    def load_dataset(self, 
                    dataset_path: str, 
                    image_col: str = "image_path",
                    text_col: str = "text_imprint",
                    label_col: str = "label") -> DataFrame:
        """
        Load multimodal dataset using Spark
        
        Args:
            dataset_path: Path to dataset
            image_col: Image column name
            text_col: Text column name
            label_col: Label column name
            
        Returns:
            Spark DataFrame
        """
        try:
            dataset_path = Path(dataset_path)
            
            if dataset_path.is_dir():
                # Load from directory structure
                df = self.image_processor.load_images_from_directory(str(dataset_path))
                
            elif dataset_path.suffix.lower() in ['.csv', '.tsv']:
                # Load from CSV
                df = self.spark.read.csv(str(dataset_path), header=True, inferSchema=True)
                
            elif dataset_path.suffix.lower() in ['.json', '.jsonl']:
                # Load from JSON
                df = self.spark.read.json(str(dataset_path))
                
            elif dataset_path.suffix.lower() == '.parquet':
                # Load from Parquet
                df = self.spark.read.parquet(str(dataset_path))
                
            else:
                raise ValueError(f"Unsupported dataset format: {dataset_path.suffix}")
            
            print(f"✅ Dataset loaded with {df.count()} samples")
            return df
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return None
    
    def preprocess_data(self, df: DataFrame) -> DataFrame:
        """
        Preprocess multimodal data using Spark
        
        Args:
            df: Raw dataset DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        try:
            # Extract image features
            print("🔄 Extracting image features...")
            df_with_img_features = self.image_processor.extract_image_features(df)
            
            # Process text features (if text column exists)
            if "text_imprint" in df.columns:
                print("🔄 Processing text features...")
                df_with_text_features = self.text_processor.process_text_imprints(df_with_img_features)
            else:
                df_with_text_features = df_with_img_features
            
            # Combine features
            print("🔄 Combining multimodal features...")
            df_processed = self._combine_features(df_with_text_features)
            
            # Remove null values
            df_processed = df_processed.dropna()
            
            print(f"✅ Data preprocessing completed. Final dataset: {df_processed.count()} samples")
            return df_processed
            
        except Exception as e:
            print(f"❌ Preprocessing error: {e}")
            return df
    
    def _combine_features(self, df: DataFrame) -> DataFrame:
        """Combine image and text features"""
        try:
            # Prepare feature columns
            feature_cols = ["features"]  # Image features
            
            if "text_features" in df.columns:
                feature_cols.append("text_features")
            
            # Vector assembler to combine features
            assembler = VectorAssembler(
                inputCols=feature_cols,
                outputCol="combined_features"
            )
            
            df_combined = assembler.transform(df)
            
            # Scale features
            scaler = StandardScaler(
                inputCol="combined_features",
                outputCol="scaled_features",
                withStd=True,
                withMean=True
            )
            
            scaler_model = scaler.fit(df_combined)
            df_scaled = scaler_model.transform(df_combined)
            
            return df_scaled
            
        except Exception as e:
            print(f"Feature combination error: {e}")
            return df
    
    def train_model(self, 
                   df: DataFrame,
                   model_type: str = "mlp",
                   validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train model using Spark ML
        
        Args:
            df: Preprocessed DataFrame
            model_type: Type of model (mlp, rf, etc.)
            validation_split: Validation split ratio
            
        Returns:
            Training results
        """
        start_time = time.time()
        
        try:
            # Split data
            train_df, val_df = df.randomSplit([1 - validation_split, validation_split], seed=42)
            
            print(f"📊 Training set: {train_df.count()} samples")
            print(f"📊 Validation set: {val_df.count()} samples")
            
            # Train model based on type
            if model_type == "mlp":
                model, pipeline = self._train_mlp(train_df)
            elif model_type == "rf":
                model, pipeline = self._train_random_forest(train_df)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Evaluate model
            predictions = model.transform(val_df)
            metrics = self._evaluate_model(predictions)
            
            # Calculate training time
            training_time = time.time() - start_time
            
            # Prepare results
            results = {
                'model': model,
                'pipeline': pipeline,
                'metrics': metrics,
                'training_time': training_time,
                'dataset_size': df.count(),
                'model_type': model_type
            }
            
            print(f"✅ Training completed in {training_time:.2f} seconds")
            print(f"📊 Validation Accuracy: {metrics['accuracy']:.4f}")
            
            return results
            
        except Exception as e:
            print(f"❌ Training error: {e}")
            return {}
    
    def _train_mlp(self, train_df: DataFrame) -> Tuple[Any, Any]:
        """Train Multi-Layer Perceptron"""
        try:
            # Get number of features
            feature_sample = train_df.select("scaled_features").first()
            feature_size = len(feature_sample[0])
            
            # Get number of classes
            num_classes = train_df.select("label").distinct().count()
            
            # Define MLP architecture
            hidden_layers = self.config.get("model", {}).get("classifier", {}).get("hidden_dims", [512, 256])
            layers = [feature_size] + hidden_layers + [num_classes]
            
            # Create MLP classifier
            mlp = MultilayerPerceptronClassifier(
                featuresCol="scaled_features",
                labelCol="label",
                predictionCol="prediction",
                layers=layers,
                blockSize=128,
                seed=42,
                maxIter=100
            )
            
            # Create pipeline
            pipeline = Pipeline(stages=[mlp])
            
            # Train model
            print("🏋️ Training MLP model...")
            model = pipeline.fit(train_df)
            
            return model, pipeline
            
        except Exception as e:
            print(f"MLP training error: {e}")
            raise
    
    def _train_random_forest(self, train_df: DataFrame) -> Tuple[Any, Any]:
        """Train Random Forest"""
        try:
            rf = RandomForestClassifier(
                featuresCol="scaled_features",
                labelCol="label",
                predictionCol="prediction",
                numTrees=100,
                maxDepth=10,
                seed=42
            )
            
            pipeline = Pipeline(stages=[rf])
            
            print("🌲 Training Random Forest model...")
            model = pipeline.fit(train_df)
            
            return model, pipeline
            
        except Exception as e:
            print(f"Random Forest training error: {e}")
            raise
    
    def _evaluate_model(self, predictions: DataFrame) -> Dict[str, float]:
        """Evaluate model predictions"""
        try:
            # Multi-class evaluator
            evaluator = MulticlassClassificationEvaluator(
                labelCol="label",
                predictionCol="prediction",
                metricName="accuracy"
            )
            
            accuracy = evaluator.evaluate(predictions)
            
            # Calculate additional metrics using collected data
            pred_and_labels = predictions.select("prediction", "label").collect()
            predictions_list = [float(row.prediction) for row in pred_and_labels]
            labels_list = [float(row.label) for row in pred_and_labels]
            
            # Use sklearn for additional metrics if available
            if SKLEARN_AVAILABLE:
                from sklearn.metrics import precision_score, recall_score, f1_score
                
                precision = precision_score(labels_list, predictions_list, average='weighted', zero_division=0)
                recall = recall_score(labels_list, predictions_list, average='weighted', zero_division=0)
                f1 = f1_score(labels_list, predictions_list, average='weighted', zero_division=0)
            else:
                precision = recall = f1 = 0.0
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
            return metrics
            
        except Exception as e:
            print(f"Evaluation error: {e}")
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
    
    def hyperparameter_tuning(self, 
                            train_df: DataFrame,
                            model_type: str = "mlp",
                            param_grid: Dict = None) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using CrossValidator
        
        Args:
            train_df: Training DataFrame
            model_type: Model type
            param_grid: Parameter grid for tuning
            
        Returns:
            Best model and results
        """
        try:
            print("🔧 Starting hyperparameter tuning...")
            
            if model_type == "mlp":
                # Create base MLP
                feature_sample = train_df.select("scaled_features").first()
                feature_size = len(feature_sample[0])
                num_classes = train_df.select("label").distinct().count()
                
                mlp = MultilayerPerceptronClassifier(
                    featuresCol="scaled_features",
                    labelCol="label",
                    predictionCol="prediction",
                    seed=42
                )
                
                # Parameter grid
                if param_grid is None:
                    param_grid = ParamGridBuilder() \
                        .addGrid(mlp.layers, [
                            [feature_size, 512, num_classes],
                            [feature_size, 512, 256, num_classes],
                            [feature_size, 1024, 512, num_classes]
                        ]) \
                        .addGrid(mlp.maxIter, [50, 100, 200]) \
                        .build()
                
                # Cross-validator
                evaluator = MulticlassClassificationEvaluator(
                    labelCol="label",
                    predictionCol="prediction",
                    metricName="accuracy"
                )
                
                cv = CrossValidator(
                    estimator=mlp,
                    estimatorParamMaps=param_grid,
                    evaluator=evaluator,
                    numFolds=3,
                    seed=42
                )
                
                # Fit cross-validator
                cv_model = cv.fit(train_df)
                
                return {
                    'best_model': cv_model.bestModel,
                    'cv_results': cv_model,
                    'best_params': cv_model.bestModel.extractParamMap()
                }
            
        except Exception as e:
            print(f"Hyperparameter tuning error: {e}")
            return {}
    
    def save_model(self, 
                  model: Any,
                  model_name: str,
                  metrics: Dict[str, float],
                  description: str = "") -> str:
        """
        Save trained model to registry
        
        Args:
            model: Trained Spark model
            model_name: Model name
            metrics: Training metrics
            description: Model description
            
        Returns:
            Model ID
        """
        try:
            # Register model
            model_id = self.model_registry.register_model(
                name=model_name,
                training_method=TrainingMethod.PYSPARK,
                model_artifact=model,
                config=self.config,
                metrics=metrics,
                description=description,
                tags=["spark", "distributed", "multimodal"]
            )
            
            return model_id
            
        except Exception as e:
            print(f"Model saving error: {e}")
            return ""
    
    def cleanup(self):
        """Clean up Spark session"""
        if self.spark:
            self.spark.stop()
            print("✅ Spark session stopped")


def create_spark_trainer(config_path: str = None) -> SparkMultimodalTrainer:
    """
    Create Spark trainer with configuration
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configured SparkMultimodalTrainer
    """
    if config_path:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default configuration
        config = {
            "data": {
                "spark": {
                    "app_name": "PillRecognitionSpark",
                    "master": "local[*]",
                    "executor_memory": "8g",
                    "driver_memory": "4g"
                }
            },
            "model": {
                "classifier": {
                    "num_classes": 1000,
                    "hidden_dims": [512, 256]
                }
            }
        }
    
    return SparkMultimodalTrainer(config)


# Training pipeline functions
def train_spark_model(dataset_path: str, 
                     config_path: str = None,
                     model_name: str = "spark_pill_model",
                     model_type: str = "mlp") -> Dict[str, Any]:
    """
    Complete Spark training pipeline
    
    Args:
        dataset_path: Path to dataset
        config_path: Configuration file path
        model_name: Name for saved model
        model_type: Type of model to train
        
    Returns:
        Training results
    """
    if not PYSPARK_AVAILABLE:
        raise ImportError("PySpark not available. Please install pyspark.")
    
    trainer = None
    try:
        # Create trainer
        trainer = create_spark_trainer(config_path)
        
        # Load dataset
        df = trainer.load_dataset(dataset_path)
        if df is None:
            raise ValueError("Failed to load dataset")
        
        # Preprocess data
        df_processed = trainer.preprocess_data(df)
        
        # Train model
        results = trainer.train_model(df_processed, model_type=model_type)
        
        if results:
            # Save model
            model_id = trainer.save_model(
                results['model'],
                model_name,
                results['metrics'],
                f"Spark {model_type} model trained on {dataset_path}"
            )
            
            results['model_id'] = model_id
            
            print(f"✅ Spark training completed successfully!")
            print(f"📊 Model ID: {model_id}")
            print(f"🎯 Accuracy: {results['metrics']['accuracy']:.4f}")
            print(f"⏱️  Training time: {results['training_time']:.2f}s")
        
        return results
        
    except Exception as e:
        print(f"❌ Spark training failed: {e}")
        return {}
    
    finally:
        if trainer:
            trainer.cleanup()


if __name__ == "__main__":
    # Example usage
    config = {
        "data": {
            "spark": {
                "app_name": "PillRecognitionTest",
                "master": "local[2]",
                "executor_memory": "4g",
                "driver_memory": "2g"
            }
        },
        "model": {
            "classifier": {
                "num_classes": 10,
                "hidden_dims": [128, 64]
            }
        }
    }
    
    print("🧪 Testing Spark trainer...")
    trainer = SparkMultimodalTrainer(config)
    trainer.cleanup()