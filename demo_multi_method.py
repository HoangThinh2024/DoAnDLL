#!/usr/bin/env python3
"""
Example Usage Script for Multi-Method Training System

This script demonstrates how to use the multi-method training system
for smart pill recognition with PyTorch, PySpark, and Transformers.

Author: DoAnDLL Team
Date: 2024
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from core.data.data_processing import DataProcessor
from core.models.model_registry import ModelRegistry, TrainingMethod
from train_multi_method import MultiMethodTrainer


def create_example_dataset():
    """Create a small example dataset for demonstration"""
    print("🔧 Creating example dataset...")
    
    config = {
        'data': {'image_size': 224},
        'model': {'text_encoder': {'max_length': 128}},
        'training': {'batch_size': 16}
    }
    
    processor = DataProcessor(config)
    
    # Create dummy dataset
    dataset_path = processor.create_dummy_dataset(
        output_path='./example_dataset',
        num_samples=100,
        num_classes=5
    )
    
    return dataset_path


def demo_pytorch_training():
    """Demonstrate PyTorch training"""
    print("\n🔥 PyTorch Training Demo")
    print("=" * 50)
    
    # Create trainer
    trainer = MultiMethodTrainer()
    
    # Create example dataset
    dataset_path = create_example_dataset()
    
    # Train PyTorch model
    results = trainer.train_single_method(
        method='pytorch',
        dataset_path=dataset_path,
        model_name='demo_pytorch_model'
    )
    
    if results:
        print(f"✅ PyTorch training completed!")
        print(f"   Model ID: {results.get('model_id', 'N/A')}")
        
    return results


def demo_spark_training():
    """Demonstrate Spark training"""
    print("\n⚡ Spark Training Demo")
    print("=" * 50)
    
    trainer = MultiMethodTrainer()
    dataset_path = create_example_dataset()
    
    # Train Spark model
    results = trainer.train_single_method(
        method='spark',
        dataset_path=dataset_path,
        model_name='demo_spark_model'
    )
    
    if results:
        print(f"✅ Spark training completed!")
        print(f"   Model ID: {results.get('model_id', 'N/A')}")
        
    return results


def demo_transformers_training():
    """Demonstrate Transformers training"""
    print("\n🤗 Transformers Training Demo") 
    print("=" * 50)
    
    trainer = MultiMethodTrainer()
    dataset_path = create_example_dataset()
    
    # Train Transformers model
    results = trainer.train_single_method(
        method='transformers',
        dataset_path=dataset_path,
        model_name='demo_transformers_model'
    )
    
    if results:
        print(f"✅ Transformers training completed!")
        print(f"   Model ID: {results.get('model_id', 'N/A')}")
        
    return results


def demo_all_methods():
    """Demonstrate training with all methods"""
    print("\n🎯 Training All Methods Demo")
    print("=" * 50)
    
    trainer = MultiMethodTrainer()
    dataset_path = create_example_dataset()
    
    # Train all methods
    all_results = trainer.train_all_methods(
        dataset_path=dataset_path,
        model_prefix='demo_all_methods'
    )
    
    return all_results


def demo_model_registry():
    """Demonstrate model registry functionality"""
    print("\n📚 Model Registry Demo")
    print("=" * 50)
    
    registry = ModelRegistry()
    
    # List all models
    print("📋 All registered models:")
    models = registry.list_models()
    
    if not models:
        print("   No models found. Run training demos first.")
        return
    
    for i, model in enumerate(models[:5], 1):  # Show first 5
        print(f"   {i}. {model['model_id']} ({model['training_method']})")
        print(f"      Accuracy: {model['accuracy']:.4f}, Created: {model['created_at'][:19]}")
    
    # Get best model
    best_model = registry.get_best_model(metric='accuracy')
    if best_model:
        print(f"\n🏆 Best model by accuracy:")
        print(f"   ID: {best_model['model_id']}")
        print(f"   Method: {best_model['training_method']}")
        print(f"   Accuracy: {best_model['accuracy']:.4f}")


def demo_comparison_report():
    """Demonstrate comparison report generation"""
    print("\n📊 Comparison Report Demo")
    print("=" * 50)
    
    registry = ModelRegistry()
    
    # Generate comparison report
    report = registry.generate_comparison_report()
    
    if 'error' in report:
        print(f"❌ {report['error']}")
        return
    
    print(f"📋 Report generated at: {report['generated_at']}")
    print(f"📊 Total models: {report['total_models']}")
    print(f"🔬 Methods compared: {', '.join(report['training_methods'])}")
    
    if 'best_performers' in report:
        best = report['best_performers']
        print(f"\n🏆 Best Performers:")
        if 'overall_accuracy' in best:
            acc_best = best['overall_accuracy']
            print(f"   🎯 Highest Accuracy: {acc_best.get('method', 'N/A')} ({acc_best.get('accuracy', 0):.4f})")
        
        if 'overall_speed' in best:
            speed_best = best['overall_speed'] 
            print(f"   ⚡ Fastest Training: {speed_best.get('method', 'N/A')} ({speed_best.get('training_time', 0):.1f}s)")


def demo_benchmark():
    """Demonstrate benchmark functionality"""
    print("\n🏁 Benchmark Demo (Quick Mode)")
    print("=" * 50)
    
    trainer = MultiMethodTrainer()
    dataset_path = create_example_dataset()
    
    # Run quick benchmark
    report = trainer.run_benchmark(
        dataset_path=dataset_path,
        output_dir='./demo_benchmark_results',
        quick_mode=True
    )
    
    if report and 'summary' in report:
        print(f"📋 Benchmark Summary: {report['summary']}")
        print(f"📁 Results saved to: ./demo_benchmark_results")


def main():
    """Run all demos"""
    print("🚀 Multi-Method Training System Demo")
    print("=" * 80)
    print("This demo showcases the three training methods and comparison capabilities.")
    print()
    
    try:
        # Demo individual methods
        print("📍 Demo 1: Individual Training Methods")
        demo_pytorch_training()
        demo_spark_training() 
        demo_transformers_training()
        
        # Demo all methods together
        print("\n📍 Demo 2: Training All Methods")
        demo_all_methods()
        
        # Demo model registry
        print("\n📍 Demo 3: Model Registry")
        demo_model_registry()
        
        # Demo comparison report
        print("\n📍 Demo 4: Comparison Report")
        demo_comparison_report()
        
        # Demo benchmark
        print("\n📍 Demo 5: Benchmark System")
        demo_benchmark()
        
        print(f"\n🎉 Demo completed successfully!")
        print(f"📁 Check the following directories for results:")
        print(f"   - ./example_dataset/ (example data)")
        print(f"   - ./models_registry/ (trained models)")
        print(f"   - ./demo_benchmark_results/ (benchmark results)")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()