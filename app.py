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

# Import custom modules
import sys
sys.path.append('src')

from models.multimodal_transformer import MultimodalPillTransformer
from data.data_processing import PillDataset, get_data_transforms
from utils.utils import load_checkpoint, get_device
from utils.metrics import MetricsCalculator

# Configure page
st.set_page_config(
    page_title="Hệ thống Nhận dạng Viên Thuốc Multimodal",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e86c1;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #2e86c1;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .prediction-result {
        background-color: #e8f5e8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ffeaa7;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_config():
    """Load model and configuration"""
    try:
        # Load configuration
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        # Initialize model
        device = get_device()
        model = MultimodalPillTransformer(config["model"])
        
        # Load checkpoint if available
        checkpoint_path = "checkpoints/best_model.pth"
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            st.success("✅ Model loaded successfully!")
        else:
            st.warning("⚠️ No trained model found. Using random weights.")
        
        model.to(device)
        model.eval()
        
        return model, config, device
    
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None, None


@st.cache_data
def load_sample_data():
    """Load sample data for demonstration"""
    try:
        # Create dummy data for demo
        pill_classes = [
            "Acetaminophen 500mg", "Ibuprofen 200mg", "Aspirin 325mg",
            "Metformin 500mg", "Lisinopril 10mg", "Atorvastatin 20mg",
            "Amlodipine 5mg", "Omeprazole 20mg", "Levothyroxine 50mcg",
            "Simvastatin 40mg"
        ]
        
        sample_data = []
        for i, pill_class in enumerate(pill_classes):
            sample_data.append({
                "id": i,
                "name": pill_class,
                "imprint": f"PILL{i:03d}",
                "description": f"Mô tả chi tiết về {pill_class}",
                "confidence": np.random.uniform(0.85, 0.99)
            })
        
        return sample_data
    
    except Exception as e:
        st.error(f"Error loading sample data: {str(e)}")
        return []


def preprocess_image(image: Image.Image, target_size: int = 224) -> torch.Tensor:
    """Preprocess image for model input"""
    try:
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image
        image = image.resize((target_size, target_size))
        
        # Convert to numpy array
        image_array = np.array(image, dtype=np.float32) / 255.0
        
        # Normalize using ImageNet stats
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_array = (image_array - mean) / std
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
        
        return image_tensor
    
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None


def predict_pill(model, image_tensor: torch.Tensor, text_imprint: str, 
                device, tokenizer, sample_data: List[Dict]) -> Dict[str, Any]:
    """Make prediction on pill image and text"""
    try:
        with torch.no_grad():
            # Move image to device
            image_tensor = image_tensor.to(device)
            
            # Tokenize text
            text_inputs = tokenizer(
                [text_imprint],
                max_length=128,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(device)
            
            # Get model predictions
            outputs = model(image_tensor, text_inputs, return_features=True)
            
            # Get probabilities
            probs = torch.softmax(outputs["logits"], dim=1)
            top_probs, top_indices = torch.topk(probs, k=5, dim=1)
            
            # Format results
            predictions = []
            for i in range(5):
                idx = top_indices[0][i].item()
                confidence = top_probs[0][i].item()
                
                # Get corresponding pill info (use sample data for demo)
                if idx < len(sample_data):
                    pill_info = sample_data[idx]
                    predictions.append({
                        "rank": i + 1,
                        "class_id": idx,
                        "name": pill_info["name"],
                        "imprint": pill_info["imprint"],
                        "confidence": confidence,
                        "description": pill_info["description"]
                    })
            
            return {
                "predictions": predictions,
                "features": {
                    "visual": outputs["visual_features"],
                    "text": outputs["text_features"],
                    "fused": outputs["fused_features"]
                }
            }
    
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None


def display_prediction_results(results: Dict[str, Any]):
    """Display prediction results"""
    if not results or "predictions" not in results:
        st.error("❌ No prediction results to display")
        return
    
    st.markdown('<div class="section-header">🎯 Kết quả Nhận dạng</div>', unsafe_allow_html=True)
    
    # Top prediction
    top_pred = results["predictions"][0]
    
    st.markdown(f"""
    <div class="prediction-result">
        <h3>🏆 Dự đoán chính: {top_pred['name']}</h3>
        <p><strong>Text Imprint:</strong> {top_pred['imprint']}</p>
        <p><strong>Độ tin cậy:</strong> {top_pred['confidence']:.2%}</p>
        <p><strong>Mô tả:</strong> {top_pred['description']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Top 5 predictions
    st.markdown('<div class="section-header">📊 Top 5 Dự đoán</div>', unsafe_allow_html=True)
    
    pred_df = pd.DataFrame([
        {
            "Thứ hạng": pred["rank"],
            "Tên thuốc": pred["name"],
            "Text Imprint": pred["imprint"],
            "Độ tin cậy": f"{pred['confidence']:.2%}"
        }
        for pred in results["predictions"]
    ])
    
    st.dataframe(pred_df, use_container_width=True)
    
    # Confidence chart
    fig = px.bar(
        x=[pred["name"][:20] + "..." if len(pred["name"]) > 20 else pred["name"] 
           for pred in results["predictions"]],
        y=[pred["confidence"] for pred in results["predictions"]],
        title="Độ tin cậy các dự đoán hàng đầu",
        labels={"x": "Loại thuốc", "y": "Độ tin cậy"}
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def display_feature_analysis(features: Dict[str, torch.Tensor]):
    """Display feature analysis"""
    st.markdown('<div class="section-header">🔍 Phân tích Features</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        visual_magnitude = torch.norm(features["visual"], dim=1).item()
        st.markdown(f"""
        <div class="metric-card">
            <h4>🖼️ Visual Features</h4>
            <p>Magnitude: {visual_magnitude:.4f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        text_magnitude = torch.norm(features["text"], dim=1).item()
        st.markdown(f"""
        <div class="metric-card">
            <h4>📝 Text Features</h4>
            <p>Magnitude: {text_magnitude:.4f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        fused_magnitude = torch.norm(features["fused"], dim=1).item()
        st.markdown(f"""
        <div class="metric-card">
            <h4>🔗 Fused Features</h4>
            <p>Magnitude: {fused_magnitude:.4f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature similarity
    visual_norm = torch.nn.functional.normalize(features["visual"], dim=1)
    text_norm = torch.nn.functional.normalize(features["text"], dim=1)
    similarity = torch.sum(visual_norm * text_norm, dim=1).item()
    
    st.markdown(f"""
    <div class="metric-card">
        <h4>🤝 Visual-Text Similarity</h4>
        <p>Cosine Similarity: {similarity:.4f}</p>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<div class="main-header">💊 Hệ thống Nhận dạng Viên Thuốc Multimodal</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Sử dụng Multimodal Transformer để nhận dạng viên thuốc từ hình ảnh và text imprint
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    with st.sidebar:
        selected = option_menu(
            "Menu chính",
            ["🏠 Trang chủ", "🔍 Nhận dạng", "📊 Thống kê", "ℹ️ Thông tin"],
            icons=['house', 'search', 'bar-chart', 'info-circle'],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "18px"},
                "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": "#1f77b4"},
            }
        )
    
    # Load model and data
    model, config, device = load_model_and_config()
    sample_data = load_sample_data()
    
    if model is None:
        st.error("❌ Không thể tải model. Vui lòng kiểm tra cài đặt.")
        return
    
    # Get tokenizer
    tokenizer = model.get_text_tokenizer()
    
    if selected == "🏠 Trang chủ":
        st.markdown('<div class="section-header">🏠 Trang chủ</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### Giới thiệu hệ thống
            
            Hệ thống nhận dạng viên thuốc sử dụng công nghệ **Multimodal Transformer** tiên tiến để:
            
            - 🖼️ **Phân tích hình ảnh viên thuốc** sử dụng Vision Transformer (ViT)
            - 📝 **Xử lý text imprint** trên viên thuốc bằng BERT
            - 🔗 **Kết hợp thông tin** từ hai nguồn dữ liệu bằng Cross-modal Attention
            - ⚡ **Xử lý song song** với Apache Spark và GPU acceleration
            
            ### Tính năng chính
            
            - ✅ Nhận dạng chính xác cao với độ tin cậy
            - ✅ Xử lý đồng thời hình ảnh và text
            - ✅ Giao diện thân thiện và dễ sử dụng
            - ✅ Hỗ trợ batch processing cho dữ liệu lớn
            """)
        
        with col2:
            st.markdown("""
            ### 📈 Thống kê hệ thống
            """)
            
            if config:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>🎯 Số lớp thuốc</h4>
                    <p>{config["model"]["classifier"]["num_classes"]} lớp</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-card">
                    <h4>🖼️ Kích thước ảnh</h4>
                    <p>{config["data"]["image_size"]}x{config["data"]["image_size"]} pixels</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-card">
                    <h4>📝 Độ dài text tối đa</h4>
                    <p>{config["model"]["text_encoder"]["max_length"]} tokens</p>
                </div>
                """, unsafe_allow_html=True)
    
    elif selected == "🔍 Nhận dạng":
        st.markdown('<div class="section-header">🔍 Nhận dạng Viên Thuốc</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 📸 Tải lên hình ảnh viên thuốc")
            uploaded_file = st.file_uploader(
                "Chọn hình ảnh...",
                type=['png', 'jpg', 'jpeg'],
                help="Hỗ trợ định dạng: PNG, JPG, JPEG"
            )
            
            if uploaded_file is not None:
                # Display uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Hình ảnh đã tải lên", use_column_width=True)
                
                # Preprocess image
                image_tensor = preprocess_image(image)
                
                if image_tensor is not None:
                    st.success("✅ Hình ảnh đã được xử lý thành công!")
        
        with col2:
            st.markdown("#### 📝 Nhập text imprint")
            text_imprint = st.text_input(
                "Text trên viên thuốc:",
                placeholder="Ví dụ: PILL123, MED500, RX10...",
                help="Nhập text được in trên viên thuốc (nếu có)"
            )
            
            st.markdown("#### ⚙️ Cài đặt")
            show_features = st.checkbox("Hiển thị phân tích features", value=True)
            confidence_threshold = st.slider(
                "Ngưỡng độ tin cậy",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Chỉ hiển thị kết quả có độ tin cậy trên ngưỡng này"
            )
        
        # Prediction button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🎯 Nhận dạng viên thuốc", type="primary", use_container_width=True):
                if uploaded_file is not None and image_tensor is not None:
                    with st.spinner("🔄 Đang phân tích..."):
                        # Add progress bar
                        progress_bar = st.progress(0)
                        for i in range(100):
                            time.sleep(0.01)
                            progress_bar.progress(i + 1)
                        
                        # Make prediction
                        results = predict_pill(
                            model, image_tensor, text_imprint or "",
                            device, tokenizer, sample_data
                        )
                        
                        if results:
                            # Filter by confidence threshold
                            filtered_predictions = [
                                pred for pred in results["predictions"]
                                if pred["confidence"] >= confidence_threshold
                            ]
                            
                            if filtered_predictions:
                                results["predictions"] = filtered_predictions
                                display_prediction_results(results)
                                
                                if show_features:
                                    display_feature_analysis(results["features"])
                            else:
                                st.warning(f"⚠️ Không có dự đoán nào đạt ngưỡng tin cậy {confidence_threshold:.1%}")
                        else:
                            st.error("❌ Có lỗi xảy ra trong quá trình nhận dạng")
                else:
                    st.warning("⚠️ Vui lòng tải lên hình ảnh trước khi nhận dạng")
    
    elif selected == "📊 Thống kê":
        st.markdown('<div class="section-header">📊 Thống kê Hệ thống</div>', unsafe_allow_html=True)
        
        # Sample statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Phân bố dữ liệu mẫu")
            
            # Create sample distribution chart
            pill_types = [pill["name"] for pill in sample_data[:5]]
            confidences = [pill["confidence"] for pill in sample_data[:5]]
            
            fig = px.pie(
                values=confidences,
                names=pill_types,
                title="Phân bố các loại thuốc mẫu"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 🎯 Hiệu suất Model")
            
            # Mock performance metrics
            metrics_data = {
                "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
                "Training": [0.95, 0.94, 0.93, 0.94],
                "Validation": [0.89, 0.88, 0.87, 0.88]
            }
            
            metrics_df = pd.DataFrame(metrics_data)
            fig = px.bar(
                metrics_df,
                x="Metric",
                y=["Training", "Validation"],
                title="Hiệu suất Model trên tập Train và Validation",
                barmode="group"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Training progress
        st.markdown("#### 📉 Quá trình Training")
        
        # Mock training data
        epochs = list(range(1, 51))
        train_loss = [0.8 * np.exp(-x/10) + 0.1 + np.random.normal(0, 0.02) for x in epochs]
        val_loss = [0.9 * np.exp(-x/12) + 0.15 + np.random.normal(0, 0.03) for x in epochs]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=train_loss, mode='lines', name='Training Loss'))
        fig.add_trace(go.Scatter(x=epochs, y=val_loss, mode='lines', name='Validation Loss'))
        fig.update_layout(
            title="Loss theo Epoch",
            xaxis_title="Epoch",
            yaxis_title="Loss"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif selected == "ℹ️ Thông tin":
        st.markdown('<div class="section-header">ℹ️ Thông tin Hệ thống</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 🏗️ Kiến trúc Hệ thống
            
            #### 🤖 Multimodal Transformer
            - **Visual Encoder**: Vision Transformer (ViT) để xử lý hình ảnh
            - **Text Encoder**: BERT để xử lý text imprint
            - **Cross-modal Attention**: Kết hợp thông tin từ hai modality
            - **Fusion Layer**: Tổng hợp features cuối cùng
            - **Classifier**: Phân loại viên thuốc
            
            #### 💾 Big Data Processing
            - **Apache Spark**: Xử lý dữ liệu phân tán
            - **Rapids cuDF/cuML**: Tăng tốc GPU
            - **Apache Parquet**: Lưu trữ dữ liệu hiệu quả
            - **Elasticsearch**: Index và tìm kiếm text
            
            #### 🚀 Tech Stack
            - **Framework**: PyTorch, Transformers, Streamlit
            - **Data**: PySpark, Pandas, NumPy
            - **Visualization**: Plotly, Matplotlib
            - **Deployment**: Docker, Kubernetes
            """)
        
        with col2:
            st.markdown("""
            ### 🔧 Cấu hình Model
            """)
            
            if config:
                with st.expander("📋 Model Configuration"):
                    st.json(config["model"])
                
                with st.expander("🎯 Training Configuration"):
                    st.json(config["training"])
                
                with st.expander("💾 Data Configuration"):
                    st.json(config["data"])
        
        st.markdown("---")
        
        st.markdown("""
        ### 👥 Nhóm phát triển
        
        - **Học viên**: [Tên sinh viên]
        - **Môn học**: Đồ án Đại học
        - **Trường**: [Tên trường]
        - **Năm**: 2025
        
        ### 📞 Liên hệ
        
        - **Email**: [email@example.com]
        - **GitHub**: [github.com/username]
        - **Website**: [website.com]
        """)


if __name__ == "__main__":
    main()
