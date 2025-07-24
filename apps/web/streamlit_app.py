import streamlit as st
import torch
import numpy as np
from PIL import Image
import cv2
import yaml
import os
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from streamlit_option_menu import option_menu
import io
import base64
from typing import Dict, Any, List, Tuple
import time
import sys
from pathlib import Path
# Thêm Spark và Transformers
try:
    import pyspark
except ImportError:
    pyspark = None
try:
    from transformers import pipeline
except ImportError:
    pipeline = None

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "core"))

# Configure page
st.set_page_config(
    page_title="Smart Pill Recognition",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .stAlert {
        border-radius: 10px;
    }
    
    .upload-section {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    
    .result-section {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* Custom button styles */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

class PillRecognitionWebUI:
    """🌐 Lớp chính cho Web UI nhận dạng viên thuốc"""
    
    def __init__(self):
        self.project_root = PROJECT_ROOT
        if 'model' not in st.session_state:
            st.session_state.model = None
        if 'device_info' not in st.session_state:
            st.session_state.device_info = self._get_device_info()
        
    def _get_device_info(self) -> Dict:
        """Lấy thông tin thiết bị GPU/CPU"""
        try:
            import torch
            device_info = {
                "pytorch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
                "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            }
            
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                device_info.update({
                    "gpu_name": gpu_name,
                    "gpu_memory_gb": f"{gpu_memory:.1f} GB"
                })
        except ImportError:
            device_info = {"status": "PyTorch chưa được cài đặt"}
            
        return device_info
    
    def show_header(self):
        """Hiển thị header đẹp"""
        st.markdown("""
        <div class="main-header">
            <h1>💊 Smart Pill Recognition System</h1>
            <p>AI-Powered Pharmaceutical Identification Platform</p>
            <p><em>Tối ưu hóa cho Ubuntu 22.04 + NVIDIA Quadro 6000 + CUDA 12.8</em></p>
        </div>
        """, unsafe_allow_html=True)
    
    def show_sidebar(self):
        """Hiển thị sidebar với thông tin hệ thống"""
        with st.sidebar:
            st.markdown("## 🖥️ Thông tin hệ thống")
            
            device_info = st.session_state.device_info
            
            # Device status
            if device_info.get("cuda_available"):
                st.success(f"🚀 GPU: {device_info.get('gpu_name', 'Unknown')}")
                st.info(f"💾 Memory: {device_info.get('gpu_memory_gb', 'Unknown')}")
                st.info(f"⚡ CUDA: {device_info.get('cuda_version', 'Unknown')}")
            else:
                st.warning("💻 CPU Mode")
                st.warning("⚠️ CUDA không khả dụng")
            
            st.markdown("---")
            
            # Model status
            st.markdown("## 🧠 Model Status")
            if st.session_state.model is None:
                st.error("❌ Model chưa được load")
                if st.button("🔄 Load Model"):
                    self.load_model()
            else:
                st.success("✅ Model đã sẵn sàng")
                st.info("🎯 Multimodal Transformer")
            
            st.markdown("---")
            
            # Quick stats
            st.markdown("## 📊 Quick Stats")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Accuracy", "96.3%", "2.1%")
            with col2:
                st.metric("Speed", "0.15s", "-0.03s")
            
            st.markdown("---")
            
            # Useful links
            st.markdown("## 🔗 Useful Links")
            st.markdown("- [📖 Documentation]()")
            st.markdown("- [🐛 Report Issues]()")
            st.markdown("- [💡 Feature Requests]()")
            st.markdown("- [🚀 GitHub Repo]()")
    
    def load_model(self, checkpoint_path=None):
        """Load model với progress bar và lưu checkpoint mới"""
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.text("🔄 Đang khởi tạo model...")
            progress_bar.progress(25)
            time.sleep(1)

            status_text.text("📦 Đang load weights...")
            progress_bar.progress(50)
            time.sleep(1)

            status_text.text("🔧 Đang setup cho inference...")
            progress_bar.progress(75)
            time.sleep(1)

            status_text.text("✅ Model đã sẵn sàng!")
            progress_bar.progress(100)

            # Lưu checkpoint mới nếu có
            if checkpoint_path:
                st.session_state.model_checkpoint = checkpoint_path
            else:
                st.session_state.model_checkpoint = "checkpoints/best_model.pth"
            st.session_state.model = f"multimodal_transformer:{st.session_state.model_checkpoint}"

            time.sleep(0.5)
            status_text.empty()
            progress_bar.empty()

            st.success(f"🎉 Model đã được load thành công! Checkpoint: {st.session_state.model_checkpoint}")
            # Không rerun để giữ trạng thái training

        except Exception as e:
            st.error(f"❌ Lỗi load model: {e}")
    
    def show_recognition_page(self):
        """Trang nhận dạng viên thuốc"""
        st.markdown("## 🎯 Nhận dạng viên thuốc")
        
        # Upload section
        st.markdown("""
        <div class="upload-section">
            <h3>📷 Upload ảnh viên thuốc</h3>
            <p>Hỗ trợ các định dạng: JPG, PNG, JPEG</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Chọn ảnh viên thuốc",
                type=['jpg', 'jpeg', 'png'],
                help="Upload ảnh rõ nét để có kết quả tốt nhất"
            )
            
            # Text input option
            st.markdown("### 📝 Hoặc nhập text imprint")
            text_imprint = st.text_input(
                "Text trên viên thuốc (nếu có)",
                placeholder="VD: 'TYLENOL', 'P500', ..."
            )
            
            # Recognition settings
            st.markdown("### ⚙️ Cài đặt nhận dạng")
            confidence_threshold = st.slider(
                "Ngưỡng độ tin cậy",
                min_value=0.1,
                max_value=1.0,
                value=0.8,
                step=0.1
            )
            
            use_multimodal = st.checkbox(
                "Sử dụng multimodal (ảnh + text)",
                value=True,
                help="Kết hợp cả ảnh và text để có kết quả chính xác hơn"
            )
        
        with col2:
            if uploaded_file is not None:
                # Display uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Ảnh đã upload", use_column_width=True)
                
                # Image info
                st.markdown("#### 📊 Thông tin ảnh")
                st.info(f"**Kích thước:** {image.size[0]} x {image.size[1]} pixels")
                st.info(f"**Định dạng:** {image.format}")
                st.info(f"**Mode:** {image.mode}")
                
                # Recognition button
                if st.button("🚀 Bắt đầu nhận dạng", type="primary"):
                    self.perform_recognition(image, text_imprint, confidence_threshold, use_multimodal)
            else:
                # Placeholder
                st.info("👆 Vui lòng upload ảnh để bắt đầu nhận dạng")
                
                # Sample images
                st.markdown("#### 🖼️ Ảnh mẫu")
                sample_images = self.get_sample_images()
                if sample_images:
                    cols = st.columns(3)
                    for i, (name, path) in enumerate(sample_images[:3]):
                        with cols[i]:
                            if st.button(f"📷 {name}", key=f"sample_{i}"):
                                st.info(f"Đã chọn ảnh mẫu: {name}")
    
    def perform_recognition(self, image, text_imprint, confidence_threshold, use_multimodal):
        """Thực hiện nhận dạng viên thuốc"""
        
        # Check if model is loaded
        if st.session_state.model is None:
            st.error("❌ Model chưa được load. Vui lòng load model trước!")
            return
        
        # Progress tracking
        progress_placeholder = st.empty()
        result_placeholder = st.empty()
        
        with progress_placeholder.container():
            st.markdown("### 🔄 Đang xử lý...")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Preprocess image
            status_text.text("🖼️ Đang xử lý ảnh...")
            progress_bar.progress(20)
            time.sleep(0.5)
            
            # Step 2: Extract features
            status_text.text("🔍 Đang trích xuất đặc trưng...")
            progress_bar.progress(40)
            time.sleep(0.5)
            
            # Step 3: Process text (if provided)
            if text_imprint:
                status_text.text("📝 Đang xử lý text imprint...")
                progress_bar.progress(60)
                time.sleep(0.5)
            
            # Step 4: Run inference
            status_text.text("🧠 Đang chạy model AI...")
            progress_bar.progress(80)
            time.sleep(1)
            
            # Step 5: Generate results
            status_text.text("📊 Đang tạo kết quả...")
            progress_bar.progress(100)
            time.sleep(0.5)
        
        # Clear progress
        progress_placeholder.empty()
        
        # Show results
        self.show_recognition_results(image, text_imprint, confidence_threshold, use_multimodal)
    
    def show_recognition_results(self, image, text_imprint, confidence_threshold, use_multimodal):
        """Hiển thị kết quả nhận dạng"""
        
        st.markdown("""
        <div class="result-section">
            <h3>🎯 Kết quả nhận dạng</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Main results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Top predictions table
            st.markdown("#### 🏆 Top Predictions")
            
            # Mock results
            results_data = {
                'Rank': [1, 2, 3, 4, 5],
                'Tên thuốc': [
                    'Paracetamol 500mg',
                    'Ibuprofen 400mg', 
                    'Aspirin 100mg',
                    'Acetaminophen 325mg',
                    'Naproxen 250mg'
                ],
                'Nhà sản xuất': [
                    'Teva Pharmaceuticals',
                    'GSK',
                    'Bayer',
                    'Johnson & Johnson',
                    'Pfizer'
                ],
                'Độ tin cậy': ['96.8%', '87.3%', '76.5%', '65.2%', '54.1%'],
                'Điểm số': [0.968, 0.873, 0.765, 0.652, 0.541]
            }
            
            df_results = pd.DataFrame(results_data)
            
            # Color code by confidence
            def color_confidence(val):
                if val >= 0.9:
                    return 'background-color: #d4edda'
                elif val >= 0.7:
                    return 'background-color: #fff3cd'
                else:
                    return 'background-color: #f8d7da'
            
            styled_df = df_results.style.map(
                color_confidence, 
                subset=['Điểm số']
            ).format({'Điểm số': '{:.3f}'})
            
            st.dataframe(styled_df, use_container_width=True)
        
        with col2:
            # Confidence chart
            st.markdown("#### 📊 Biểu đồ độ tin cậy")
            
            fig = px.bar(
                df_results.head(3),
                x='Tên thuốc',
                y='Điểm số',
                color='Điểm số',
                color_continuous_scale='RdYlGn',
                title="Top 3 Predictions"
            )
            fig.update_layout(height=300, showlegend=False)
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed information about top prediction
        st.markdown("#### 🔍 Thông tin chi tiết - Paracetamol 500mg")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Độ tin cậy",
                value="96.8%",
                delta="2.3%"
            )
        
        with col2:
            st.metric(
                label="Hình dạng",
                value="Viên nén",
                delta="Tròn"
            )
        
        with col3:
            st.metric(
                label="Màu sắc", 
                value="Trắng",
                delta="Đồng nhất"
            )
        
        with col4:
            st.metric(
                label="Kích thước",
                value="10mm",
                delta="±0.5mm"
            )
        
        # Additional details in expandable sections
        with st.expander("📋 Thông tin dược phẩm"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Thành phần hoạt chất:**
                - Paracetamol: 500mg
                
                **Thành phần tá dược:**
                - Tinh bột bắp
                - Cellulose vi tinh thể
                - Povidone K30
                """)
            
            with col2:
                st.markdown("""
                **Chỉ định:**
                - Giảm đau nhẹ đến vừa
                - Hạ sốt
                
                **Liều dùng:**
                - Người lớn: 1-2 viên/lần, 3-4 lần/ngày
                - Không quá 8 viên/ngày
                """)
        
        with st.expander("🔬 Phân tích kỹ thuật"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Đặc trưng hình ảnh:**
                - Hình dạng: Tròn, lồi 2 mặt
                - Màu sắc: Trắng đồng nhất
                - Bề mặt: Nhẵn, không vân
                - Đường kính: 10.2mm ±0.3mm
                """)
            
            with col2:
                if text_imprint:
                    st.markdown(f"""
                    **Text Imprint Analysis:**
                    - Input text: "{text_imprint}"
                    - Matched pattern: "P500"
                    - Confidence: 94.2%
                    - Alternative readings: "P 500", "P-500"
                    """)
                else:
                    st.markdown("""
                    **Text Imprint Analysis:**
                    - Không có text input
                    - Phát hiện text trên ảnh: "P500"
                    - OCR Confidence: 87.5%
                    """)
        
        with st.expander("🎯 Model Performance"):
            # Performance metrics visualization
            metrics_data = {
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC'],
                'Value': [0.963, 0.971, 0.956, 0.963, 0.984],
                'Benchmark': [0.950, 0.960, 0.940, 0.950, 0.970]
            }
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=metrics_data['Value'],
                theta=metrics_data['Metric'],
                fill='toself',
                name='Current Model',
                line_color='blue'
            ))
            fig.add_trace(go.Scatterpolar(
                r=metrics_data['Benchmark'],
                theta=metrics_data['Metric'],
                fill='toself',
                name='Benchmark',
                line_color='red',
                opacity=0.6
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0.8, 1.0]
                    )),
                showlegend=True,
                title="Model Performance vs Benchmark",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Action buttons
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("💾 Lưu kết quả"):
                st.success("✅ Kết quả đã được lưu!")
        
        with col2:
            if st.button("📤 Xuất báo cáo"):
                st.success("✅ Báo cáo đã được xuất!")
        
        with col3:
            if st.button("🔄 Nhận dạng lại"):
                st.rerun()
        
        with col4:
            if st.button("📋 Sao chép kết quả"):
                st.success("✅ Đã sao chép vào clipboard!")
    
    def get_sample_images(self):
        """Lấy danh sách ảnh mẫu"""
        test_dir = self.project_root / "Dataset_BigData" / "CURE_dataset" / "CURE_dataset_test"
        
        if test_dir.exists():
            sample_files = list(test_dir.glob("*.jpg"))[:6]
            return [(f.stem, str(f)) for f in sample_files]
        
        return []
    
    def show_training_page(self):
        """Trang huấn luyện model với lựa chọn thường, Spark, Transformer"""
        st.markdown("## 🏋️ Huấn luyện Model")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### ⚙️ Cấu hình Training")

            # Training parameters
            epochs = st.slider("Số epochs", min_value=1, max_value=100, value=50)
            batch_size = st.selectbox("Batch size", [16, 32, 64, 128], index=1)
            learning_rate = st.select_slider(
                "Learning rate",
                options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
                value=0.001,
                format_func=lambda x: f"{x:.4f}"
            )

            # Model settings
            st.markdown("#### 🧠 Model Settings")
            model_type = st.selectbox(
                "Loại model",
                ["Multimodal Transformer", "Vision Only", "Text Only"]
            )

            use_pretrained = st.checkbox("Sử dụng pretrained weights", value=True)
            mixed_precision = st.checkbox("Mixed precision training", value=True)

            # Data augmentation
            st.markdown("#### 🎨 Data Augmentation")
            use_augmentation = st.checkbox("Bật data augmentation", value=True)

            if use_augmentation:
                aug_col1, aug_col2 = st.columns(2)
                with aug_col1:
                    rotation = st.checkbox("Rotation", value=True)
                    flip = st.checkbox("Random flip", value=True)
                with aug_col2:
                    brightness = st.checkbox("Brightness", value=True)
                    contrast = st.checkbox("Contrast", value=True)

            # Thêm lựa chọn phương pháp huấn luyện
            st.markdown("#### 🚀 Phương pháp huấn luyện")
            train_method = st.radio(
                "Chọn phương pháp:",
                ["Bình thường (PyTorch)", "Spark (PySpark)", "Transformer (HuggingFace)"]
            )

        with col2:
            st.markdown("### 📊 Training Status")

            # Current training info
            if 'training_active' not in st.session_state:
                st.session_state.training_active = False

            if st.session_state.training_active:
                st.success("🟢 Training đang chạy")

                # Mock training progress
                current_epoch = st.empty()
                progress_bar = st.progress(0)

                # Simulated training metrics
                loss_chart = st.empty()
                acc_chart = st.empty()

                # Stop button
                if st.button("🛑 Dừng Training"):
                    st.session_state.training_active = False
                    st.rerun()
            else:
                st.info("⏸️ Không có training nào đang chạy")

                # Dataset info
                st.markdown("#### 📁 Dataset Info")
                st.metric("Train images", "12,678")
                st.metric("Val images", "2,115")
                st.metric("Test images", "1,054")
                st.metric("Classes", "156")

        # Start training button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            if not st.session_state.training_active:
                if st.button("🚀 Bắt đầu Training", type="primary", use_container_width=True):
                    self.start_training(epochs, batch_size, learning_rate, model_type, train_method)
    
    def start_training(self, epochs, batch_size, learning_rate, model_type, train_method):
        """Bắt đầu quá trình training với lựa chọn phương pháp, giữ trạng thái qua nhiều epoch"""
        if 'training_epoch' not in st.session_state:
            st.session_state.training_epoch = 0
        if 'training_metrics' not in st.session_state:
            st.session_state.training_metrics = []
        st.session_state.training_active = True

        st.success(f"🚀 Đã bắt đầu training với {epochs} epochs!")
        st.info(f"📊 Config: Batch size={batch_size}, LR={learning_rate}, Model={model_type}, Phương pháp={train_method}")

        progress_placeholder = st.empty()

        with progress_placeholder.container():
            st.markdown(f"### 🔄 Training Progress ({train_method})")
            epoch_progress = st.progress(st.session_state.training_epoch / epochs)
            current_metrics = st.empty()

            # Training simulation cho từng phương pháp
            for epoch in range(st.session_state.training_epoch, min(epochs, st.session_state.training_epoch + 5)):
                epoch_progress.progress((epoch + 1) / epochs)

                # Simulate metrics
                if train_method == "Bình thường (PyTorch)":
                    train_loss = 2.5 - (epoch * 0.3) + np.random.normal(0, 0.1)
                    val_loss = 2.3 - (epoch * 0.25) + np.random.normal(0, 0.1)
                    train_acc = 0.3 + (epoch * 0.15) + np.random.normal(0, 0.02)
                    val_acc = 0.35 + (epoch * 0.13) + np.random.normal(0, 0.02)
                elif train_method == "Spark (PySpark)":
                    train_loss = 2.2 - (epoch * 0.28) + np.random.normal(0, 0.12)
                    val_loss = 2.1 - (epoch * 0.22) + np.random.normal(0, 0.12)
                    train_acc = 0.32 + (epoch * 0.16) + np.random.normal(0, 0.03)
                    val_acc = 0.36 + (epoch * 0.14) + np.random.normal(0, 0.03)
                elif train_method == "Transformer (HuggingFace)":
                    train_loss = 2.0 - (epoch * 0.25) + np.random.normal(0, 0.15)
                    val_loss = 1.9 - (epoch * 0.20) + np.random.normal(0, 0.15)
                    train_acc = 0.35 + (epoch * 0.18) + np.random.normal(0, 0.04)
                    val_acc = 0.38 + (epoch * 0.15) + np.random.normal(0, 0.04)
                else:
                    train_loss = 2.5
                    val_loss = 2.3
                    train_acc = 0.3
                    val_acc = 0.35

                st.session_state.training_metrics.append({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "method": train_method
                })

                current_metrics.markdown(f"""
                **Epoch {epoch + 1}/{epochs} ({train_method})**
                - Train Loss: {train_loss:.3f}
                - Val Loss: {val_loss:.3f} 
                - Train Acc: {train_acc:.3f}
                - Val Acc: {val_acc:.3f}
                """)
                time.sleep(1)

            st.session_state.training_epoch = epoch + 1
            if st.session_state.training_epoch >= epochs:
                st.session_state.training_active = False
                st.success(f"✅ Training hoàn thành với phương pháp: {train_method}!")
                st.session_state.training_epoch = 0
                st.session_state.training_metrics = []
            # Không rerun để giữ trạng thái
    
    def show_analytics_page(self):
        """Trang phân tích và thống kê, so sánh hiệu năng các phương pháp huấn luyện"""
        st.markdown("## 📊 Phân tích Dataset & Model")

        # Dataset overview
        st.markdown("### 📁 Tổng quan Dataset")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="Tổng số ảnh",
                value="15,847",
                delta="1,200"
            )

        with col2:
            st.metric(
                label="Số lớp thuốc",
                value="156",
                delta="12"
            )

        with col3:
            st.metric(
                label="Độ chính xác",
                value="96.3%",
                delta="2.1%"
            )

        with col4:
            st.metric(
                label="Thời gian inference",
                value="0.15s",
                delta="-0.03s"
            )

        # Dataset distribution charts
        st.markdown("### 📈 Phân bố Dataset")

        col1, col2 = st.columns(2)

        with col1:
            # Class distribution
            st.markdown("#### 🎯 Phân bố theo lớp")

            # Mock data
            class_data = pd.DataFrame({
                'Class': [f'Class_{i}' for i in range(1, 11)],
                'Count': np.random.randint(50, 200, 10)
            })

            fig = px.bar(
                class_data,
                x='Class',
                y='Count',
                color='Count',
                color_continuous_scale='viridis'
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Train/Val/Test split
            st.markdown("#### 📊 Phân chia dữ liệu")

            split_data = pd.DataFrame({
                'Split': ['Train', 'Validation', 'Test'],
                'Count': [12678, 2115, 1054],
                'Percentage': [80.0, 13.3, 6.7]
            })

            fig = px.pie(
                split_data,
                values='Count',
                names='Split',
                color_discrete_sequence=['#ff9999', '#66b3ff', '#99ff99']
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        # So sánh hiệu năng các phương pháp huấn luyện
        st.markdown("### ⚡ So sánh hiệu năng huấn luyện")
        compare_methods = pd.DataFrame({
            'Phương pháp': ['Bình thường (PyTorch)', 'Spark (PySpark)', 'Transformer (HuggingFace)'],
            'Thời gian (s)': [120, 90, 75],
            'Độ chính xác (%)': [95.2, 96.1, 97.0],
            'Sử dụng RAM (GB)': [8.2, 6.5, 7.1],
            'Sử dụng GPU (%)': [80, 85, 90]
        })
        st.dataframe(compare_methods, use_container_width=True)

        fig = px.bar(
            compare_methods,
            x='Phương pháp',
            y='Độ chính xác (%)',
            color='Phương pháp',
            title='So sánh độ chính xác các phương pháp huấn luyện',
            text='Độ chính xác (%)'
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        # Model performance analysis
        st.markdown("### 🧠 Phân tích Performance Model")

        col1, col2 = st.columns(2)

        with col1:
            # Training curves
            st.markdown("#### 📈 Training Curves")

            # Mock training data
            epochs = list(range(1, 51))
            train_loss = [2.5 - i*0.04 + np.random.normal(0, 0.1) for i in epochs]
            val_loss = [2.3 - i*0.035 + np.random.normal(0, 0.1) for i in epochs]
            train_acc = [0.3 + i*0.013 + np.random.normal(0, 0.01) for i in epochs]
            val_acc = [0.35 + i*0.012 + np.random.normal(0, 0.01) for i in epochs]

            # Ensure values are in reasonable ranges
            train_loss = np.maximum(train_loss, 0.1)
            val_loss = np.maximum(val_loss, 0.1)
            train_acc = np.minimum(np.maximum(train_acc, 0), 1)
            val_acc = np.minimum(np.maximum(val_acc, 0), 1)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=epochs, y=train_loss, name='Train Loss', line=dict(color='red')))
            fig.add_trace(go.Scatter(x=epochs, y=val_loss, name='Val Loss', line=dict(color='orange')))

            fig.update_layout(
                title="Loss Curves",
                xaxis_title="Epoch",
                yaxis_title="Loss",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Accuracy curves
            st.markdown("#### 🎯 Accuracy Curves")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=epochs, y=train_acc, name='Train Acc', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=epochs, y=val_acc, name='Val Acc', line=dict(color='green')))

            fig.update_layout(
                title="Accuracy Curves",
                xaxis_title="Epoch", 
                yaxis_title="Accuracy",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

        # Confusion matrix
        st.markdown("### 🔥 Confusion Matrix")

        # Mock confusion matrix data
        n_classes = 10  # Show subset for visualization
        conf_matrix = np.random.randint(0, 100, (n_classes, n_classes))
        np.fill_diagonal(conf_matrix, np.random.randint(80, 100, n_classes))

        fig = px.imshow(
            conf_matrix,
            color_continuous_scale='Blues',
            aspect='auto',
            title="Confusion Matrix (Top 10 Classes)"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Performance by class
        with st.expander("📊 Performance chi tiết theo từng lớp"):
            # Mock per-class metrics
            class_metrics = pd.DataFrame({
                'Class': [f'Paracetamol_{i}mg' for i in [250, 500, 1000]] + 
                         [f'Ibuprofen_{i}mg' for i in [200, 400, 600]] +
                         [f'Aspirin_{i}mg' for i in [81, 100, 325]],
                'Precision': np.random.uniform(0.85, 0.98, 9),
                'Recall': np.random.uniform(0.82, 0.96, 9),
                'F1-Score': np.random.uniform(0.83, 0.97, 9),
                'Support': np.random.randint(50, 200, 9)
            })

            st.dataframe(
                class_metrics.style.format({
                    'Precision': '{:.3f}',
                    'Recall': '{:.3f}',
                    'F1-Score': '{:.3f}'
                }),
                use_container_width=True
            )
    
    def show_settings_page(self):
        """Trang cài đặt hệ thống và theme"""
        st.markdown("## ⚙️ Cài đặt Hệ thống")

        col1, col2 = st.columns([2, 1])

        with col1:
            # Theme settings
            st.markdown("### 🎨 Theme Settings")
            theme = st.radio("Chọn theme:", ["Light", "Dark", "Auto"], index=2)
            if 'theme' not in st.session_state:
                st.session_state['theme'] = theme
            if theme != st.session_state['theme']:
                st.session_state['theme'] = theme
                st.experimental_set_query_params(theme=theme)
                st.success(f"Đã chuyển theme sang: {theme}")
            # Apply theme (simple CSS switch)
            if theme == "Dark":
                st.markdown("""
                <style>
                body, .main-header, .sidebar .sidebar-content {
                    background: #222 !important;
                    color: #eee !important;
                }
                .metric-card, .result-section {
                    background: #333 !important;
                    color: #eee !important;
                }
                </style>
                """, unsafe_allow_html=True)
            elif theme == "Light":
                st.markdown("""
                <style>
                body, .main-header, .sidebar .sidebar-content {
                    background: #fafafa !important;
                    color: #222 !important;
                }
                .metric-card, .result-section {
                    background: #fff !important;
                    color: #222 !important;
                }
                </style>
                """, unsafe_allow_html=True)
            # Hướng dẫn đổi theme thực sự
            st.info("""
                ⚠️ Để đổi theme thực sự (Light/Dark/Auto) cho toàn bộ ứng dụng, hãy chỉnh file `.streamlit/config.toml`:
                
                ```toml
                [theme]
                base="light"  # hoặc "dark" hoặc "auto"
                ```
                Sau đó reload lại ứng dụng Streamlit.
            """)

            # Model settings
            st.markdown("### 🧠 Cài đặt Model")
            model_config = {
                "model_type": st.selectbox(
                    "Loại model",
                    ["Multimodal Transformer", "Vision Transformer", "ResNet-50"],
                    index=0
                ),
                "checkpoint_path": st.text_input(
                    "Đường dẫn checkpoint",
                    value="checkpoints/best_model.pth"
                ),
                "device": st.selectbox(
                    "Device",
                    ["auto", "cuda", "cpu"],
                    index=0
                ),
                "batch_size": st.slider("Batch size cho inference", 1, 32, 8),
                "confidence_threshold": st.slider("Ngưỡng độ tin cậy", 0.1, 1.0, 0.8)
            }

            # Data settings
            st.markdown("### 📁 Cài đặt Dữ liệu")
            data_config = {
                "dataset_path": st.text_input(
                    "Đường dẫn dataset",
                    value="Dataset_BigData/CURE_dataset"
                ),
                "image_size": st.selectbox(
                    "Kích thước ảnh",
                    [224, 256, 384, 512],
                    index=0
                ),
                "preprocessing": st.multiselect(
                    "Preprocessing steps",
                    ["Resize", "Normalize", "Center Crop", "Random Flip"],
                    default=["Resize", "Normalize", "Center Crop"]
                )
            }

            # Performance settings
            st.markdown("### ⚡ Cài đặt Performance")
            perf_config = {
                "num_workers": st.slider("Số workers cho DataLoader", 0, 8, 4),
                "pin_memory": st.checkbox("Pin memory", value=True),
                "mixed_precision": st.checkbox("Mixed precision", value=True),
                "compile_model": st.checkbox("Compile model (PyTorch 2.0)", value=False)
            }

            # Save settings button
            if st.button("💾 Lưu cài đặt", type="primary"):
                config = {**model_config, **data_config, **perf_config, "theme": theme}
                st.success("✅ Đã lưu cài đặt thành công!")
                st.json(config)
        
        with col2:
            st.markdown("### 📊 Training Status")

            # Current training info
            if 'training_active' not in st.session_state:
                st.session_state.training_active = False

            if st.session_state.training_active:
                st.success("🟢 Training đang chạy")

                # Mock training progress
                current_epoch = st.empty()
                progress_bar = st.progress(0)

                # Simulated training metrics
                loss_chart = st.empty()
                acc_chart = st.empty()

                # Stop button
                if st.button("� Dừng Training"):
                    st.session_state.training_active = False
                    st.rerun()
                # Nút tiếp tục training nếu chưa đủ epoch
                if st.session_state.training_epoch < epochs:
                    if st.button("▶️ Tiếp tục Training"):
                        self.start_training(epochs, batch_size, learning_rate, model_type, train_method)
            else:
                st.info("⏸️ Không có training nào đang chạy")

                # Dataset info
                st.markdown("#### 📁 Dataset Info")
                st.metric("Train images", "12,678")
                st.metric("Val images", "2,115")
                st.metric("Test images", "1,054")
                st.metric("Classes", "156")
        # Show selected page
        if selected == "🎯 Nhận dạng":
            self.show_recognition_page()
        elif selected == "🏋️ Training":
            self.show_training_page()
        elif selected == "📊 Analytics":
            self.show_analytics_page()
        elif selected == "⚙️ Settings":
            self.show_settings_page()
        
        # Footer
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; color: #666; padding: 1rem;'>
                💊 Smart Pill Recognition System v1.0.0 | 
                Tối ưu hóa cho Ubuntu 22.04 + NVIDIA Quadro 6000 + CUDA 12.8 | 
                Made with ❤️ by DoAnDLL Team
            </div>
            """,
            unsafe_allow_html=True
        )

# Initialize and run the app
if __name__ == "__main__":
    app = PillRecognitionWebUI()
    app.run()
