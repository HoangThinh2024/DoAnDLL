#!/usr/bin/env python3

"""
Simple Demo for CURE Dataset Integration and Port Management
"""

import os
import sys
from pathlib import Path

def check_dataset():
    """Check CURE dataset structure"""
    print("🔍 Checking CURE Dataset...")
    
    dataset_path = Path("Dataset_BigData/CURE_dataset")
    
    if not dataset_path.exists():
        print("❌ Dataset not found at Dataset_BigData/CURE_dataset")
        return False
    
    print("✅ Dataset directory found")
    
    # Check subdirectories
    train_path = dataset_path / "CURE_dataset_train_cut_bounding_box"
    val_path = dataset_path / "CURE_dataset_validation_cut_bounding_box"
    test_path = dataset_path / "CURE_dataset_test"
    
    results = {}
    
    if train_path.exists():
        train_classes = list(train_path.iterdir())
        train_classes = [d for d in train_classes if d.is_dir()]
        print(f"✅ Training data: {len(train_classes)} classes")
        
        # Count samples in first class
        if train_classes:
            first_class = train_classes[0]
            sample_count = 0
            for view in ["top", "bottom"]:
                view_dir = first_class / view
                if view_dir.exists():
                    customer_dir = view_dir / "Customer"
                    search_dir = customer_dir if customer_dir.exists() else view_dir
                    
                    for pattern in ["*.png", "*.jpg"]:
                        sample_count += len(list(search_dir.glob(pattern)))
            
            print(f"   • Sample from class {first_class.name}: {sample_count} images")
        results["train"] = len(train_classes)
    else:
        print("❌ Training data not found")
        results["train"] = 0
    
    if val_path.exists():
        val_classes = list(val_path.iterdir())
        val_classes = [d for d in val_classes if d.is_dir()]
        print(f"✅ Validation data: {len(val_classes)} classes")
        results["validation"] = len(val_classes)
    else:
        print("❌ Validation data not found")
        results["validation"] = 0
    
    if test_path.exists():
        test_images = list(test_path.glob("*.jpg")) + list(test_path.glob("*.png"))
        print(f"✅ Test data: {len(test_images)} images")
        results["test"] = len(test_images)
    else:
        print("❌ Test data not found")
        results["test"] = 0
    
    return results

def check_ports():
    """Check port availability"""
    print("\n🌐 Checking Port Availability...")
    
    import socket
    
    ports_to_check = [8088, 8051, 8501, 8502, 8503, 8504, 8505]
    available_ports = []
    
    for port in ports_to_check:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex(('localhost', port))
                if result != 0:
                    print(f"✅ Port {port}: Available")
                    available_ports.append(port)
                else:
                    print(f"❌ Port {port}: In use")
        except Exception as e:
            print(f"⚠️  Port {port}: Error checking ({e})")
    
    print(f"\n📊 Available ports: {available_ports}")
    
    # Recommendations for restricted ports
    if 8088 in available_ports:
        print("⚠️  Port 8088 is available but may be restricted on some servers")
    if 8051 in available_ports:
        print("⚠️  Port 8051 is available but may be restricted on some servers")
    
    recommended_port = 8501
    if 8501 in available_ports:
        print(f"✅ Recommended port: {recommended_port}")
    else:
        for port in available_ports:
            if port >= 8500:
                recommended_port = port
                print(f"✅ Alternative recommended port: {recommended_port}")
                break
    
    return available_ports, recommended_port

def check_dependencies():
    """Check Python dependencies"""
    print("\n📦 Checking Dependencies...")
    
    dependencies = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"), 
        ("PIL", "Pillow"),
        ("streamlit", "Streamlit"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy")
    ]
    
    missing = []
    available = []
    
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✅ {name}: Available")
            available.append(name)
        except ImportError:
            print(f"❌ {name}: Missing")
            missing.append(name)
    
    if missing:
        print(f"\n📋 Missing dependencies: {', '.join(missing)}")
        print("   Run: pip install -r requirements.txt")
    else:
        print("\n✅ All dependencies available")
    
    return len(missing) == 0

def create_simple_app():
    """Create a simple app file for testing"""
    print("\n🚀 Creating Simple Test App...")
    
    app_content = '''
import streamlit as st
import os
from pathlib import Path

st.title("💊 Smart Pill Recognition - Dataset Test")

# Check dataset
dataset_path = Path("Dataset_BigData/CURE_dataset")
if dataset_path.exists():
    st.success("✅ CURE Dataset found!")
    
    # Show dataset structure
    train_path = dataset_path / "CURE_dataset_train_cut_bounding_box"
    if train_path.exists():
        classes = [d.name for d in train_path.iterdir() if d.is_dir()]
        st.write(f"**Training Classes:** {len(classes)}")
        st.write(f"Classes: {', '.join(sorted(classes))}")
        
        # Show sample from first class
        if classes:
            first_class_path = train_path / classes[0] / "top"
            if first_class_path.exists():
                customer_path = first_class_path / "Customer"
                search_path = customer_path if customer_path.exists() else first_class_path
                
                images = list(search_path.glob("*.png")) + list(search_path.glob("*.jpg"))
                st.write(f"**Sample images in class {classes[0]}:** {len(images)}")
                
                if images:
                    try:
                        from PIL import Image
                        img = Image.open(images[0])
                        st.image(img, caption=f"Sample: {images[0].name}", width=300)
                    except:
                        st.write(f"Found image: {images[0].name}")
else:
    st.error("❌ CURE Dataset not found")
    st.info("Please place the dataset in: Dataset_BigData/CURE_dataset/")

# Port info
st.subheader("🌐 Port Information")
import socket
port = st.get_option("server.port") or 8501
st.write(f"**Current Port:** {port}")

# Test other ports
test_ports = [8088, 8051, 8502, 8503]
available = []
for p in test_ports:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            if sock.connect_ex(('localhost', p)) != 0:
                available.append(p)
    except:
        pass

st.write(f"**Other Available Ports:** {available}")

if 8088 in available:
    st.warning("⚠️ Port 8088 available but may be restricted")
if 8051 in available:
    st.warning("⚠️ Port 8051 available but may be restricted")
'''
    
    with open("app_simple.py", "w") as f:
        f.write(app_content)
    
    print("✅ Simple test app created: app_simple.py")
    print("   Run with: streamlit run app_simple.py")

def main():
    """Main function"""
    print("🧪 Smart Pill Recognition - System Check")
    print("=" * 50)
    
    # Check dataset
    dataset_results = check_dataset()
    
    # Check ports
    available_ports, recommended_port = check_ports()
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Create simple app
    create_simple_app()
    
    # Summary
    print("\n📊 Summary")
    print("=" * 20)
    
    if dataset_results:
        total_classes = dataset_results.get("train", 0) + dataset_results.get("validation", 0)
        print(f"✅ Dataset: {total_classes} total classes, {dataset_results.get('test', 0)} test images")
    else:
        print("❌ Dataset: Not available")
    
    print(f"✅ Ports: {len(available_ports)} available, recommended: {recommended_port}")
    
    if deps_ok:
        print("✅ Dependencies: All available")
    else:
        print("❌ Dependencies: Some missing")
    
    # Next steps
    print("\n🚀 Next Steps:")
    
    if dataset_results and deps_ok:
        print("1. ✅ System ready!")
        print(f"2. Run: streamlit run app_simple.py --server.port {recommended_port}")
        print("3. Or run: ./run")
    else:
        if not dataset_results:
            print("1. Ensure CURE dataset is in Dataset_BigData/CURE_dataset/")
        if not deps_ok:
            print("2. Run: pip install -r requirements.txt")
        print("3. Run this script again")

if __name__ == "__main__":
    main()
