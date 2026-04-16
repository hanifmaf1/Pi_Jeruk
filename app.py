import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import cv2
import numpy as np

# Set page config for a premium feel
st.set_page_config(
    page_title="Jeruk Insight | Citrus Quality AI",
    page_icon="🍊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for the "Wow" factor
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    .main {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
        color: #ffffff;
    }

    .stApp {
        background: transparent;
    }

    /* Glassmorphism sidebar */
    [data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Custom Header */
    .view-header {
        background: linear-gradient(90deg, #ff8c00 0%, #ffa500 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }

    .subtitle {
        color: #b0b0c0;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }

    /* Cards */
    .result-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
    }

    .result-card:hover {
        transform: translateY(-5px);
        background: rgba(255, 255, 255, 0.05);
    }

    /* Success metrics */
    .metric-container {
        display: flex;
        justify-content: space-around;
        margin-top: 1rem;
    }

    .metric-box {
        text-align: center;
        background: rgba(255, 140, 0, 0.1);
        border: 1px solid #ff8c00;
        border-radius: 10px;
        padding: 10px 20px;
        min-width: 120px;
    }

    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #ff8c00;
    }

    .metric-label {
        font-size: 0.8rem;
        color: #e0e0e0;
        text-transform: uppercase;
    }

    /* Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .fade-in {
        animation: fadeIn 0.8s ease-out forwards;
    }

    /* File uploader styling */
    .stFileUploader section {
        background-color: rgba(255, 255, 255, 0.02) !important;
        border: 2px dashed rgba(255, 140, 0, 0.3) !important;
        border-radius: 15px !important;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.markdown('<div class="view-header">Jeruk Insight</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Advanced AI for Automatic Citrus Grading & Quality Assessment</div>', unsafe_allow_html=True)

# Constants & Model Path
MODEL_PATH = 'content/runs/detect/train/weights/best.pt'
CLASS_NAMES = ['jeruk busuk', 'jeruk matang besar', 'jeruk matang sedang', 'jeruk mentah']
CLASS_COLORS = {
    'jeruk busuk': '#ff4b4b',
    'jeruk matang besar': '#00cc66',
    'jeruk matang sedang': '#ffff66',
    'jeruk mentah': '#3399ff'
}

# Sidebar - Configuration
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1728/1728765.png", width=100)
    st.title("Settings")
    st.markdown("---")
    
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    iou_threshold = st.slider("IOU Threshold", 0.0, 1.0, 0.45, 0.05)
    
    st.markdown("---")
    st.markdown("### Model Information")
    st.info(f"Model: YOLOv9 (Trained)\nWeights: {os.path.basename(MODEL_PATH)}")
    
    st.markdown("---")
    st.markdown("### Classes")
    for cls in CLASS_NAMES:
        st.markdown(f"🟠 **{cls.title()}**")

# Model Loading Helper
@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        return None
    return YOLO(path)

# Main Application Logic
def main():
    model = load_model(MODEL_PATH)
    
    if model is None:
        st.error(f"⚠️ Model file not found at `{MODEL_PATH}`. Please ensure the training process has completed and the weights are in the correct directory.")
        return

    # Image Upload Section
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader("Choose a citrus image...", type=["jpg", "jpeg", "png"])
        st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Inference
        with st.spinner("Analyzing image..."):
            results = model.predict(source=image, conf=conf_threshold, iou=iou_threshold)
            
            # Annotated image
            annotated_img = results[0].plot()
            annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            
            # Result counts
            boxes = results[0].boxes
            class_counts = {}
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = CLASS_NAMES[cls_id]
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

        with col2:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.subheader("🎯 Analysis Results")
            st.image(annotated_img_rgb, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Summary Metrics
        st.markdown('<div class="fade-in">', unsafe_allow_html=True)
        st.subheader("📊 Summary Statistics")
        
        if class_counts:
            cols = st.columns(len(class_counts))
            for i, (name, count) in enumerate(class_counts.items()):
                with cols[i]:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-value">{count}</div>
                        <div class="metric-label">{name}</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("No oranges detected in the image.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Detailed results table
        with st.expander("🔍 View Raw Detection Data"):
            if len(boxes) > 0:
                data = {
                    "Class": [CLASS_NAMES[int(box.cls[0])] for box in boxes],
                    "Confidence": [f"{float(box.conf[0]):.2%}" for box in boxes]
                }
                st.table(data)
            else:
                st.write("No detection data.")
                
    else:
        # Default view / instructions
        st.info("👋 Welcome! Please upload an image of oranges to start the automatic grading process.")
        
        # Placeholder for demonstration (optional)
        # st.image("assets/demo_placeholder.jpg", caption="Example Detection", use_container_width=True)

if __name__ == "__main__":
    main()
