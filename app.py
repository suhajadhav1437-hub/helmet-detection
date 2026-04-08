import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import os
import time

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Helmet Detection System",
    page_icon="⛑️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Inter:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark industrial theme */
    .stApp {
        background: #0a0a0f;
        color: #e0e0e0;
    }

    /* Header */
    .main-header {
        font-family: 'Rajdhani', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        letter-spacing: 4px;
        text-transform: uppercase;
        background: linear-gradient(135deg, #ff6b35, #f7c59f, #ff6b35);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1.5rem 0 0.2rem 0;
        margin: 0;
    }

    .sub-header {
        text-align: center;
        color: #666;
        font-size: 0.85rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 2rem;
        font-weight: 300;
    }

    /* Metric cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin: 1.5rem 0;
    }

    .metric-card {
        background: linear-gradient(135deg, #141418, #1a1a22);
        border: 1px solid #2a2a35;
        border-radius: 8px;
        padding: 1.2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, #ff6b35, #f7c59f);
    }

    .metric-value {
        font-family: 'Rajdhani', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #ff6b35;
        line-height: 1;
    }

    .metric-label {
        font-size: 0.7rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #555;
        margin-top: 4px;
    }

    /* Alert boxes */
    .alert-danger {
        background: linear-gradient(135deg, rgba(220,50,50,0.15), rgba(220,50,50,0.05));
        border: 1px solid rgba(220,50,50,0.4);
        border-left: 4px solid #dc3232;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
        color: #ff8080;
        font-size: 0.95rem;
    }

    .alert-success {
        background: linear-gradient(135deg, rgba(50,200,100,0.15), rgba(50,200,100,0.05));
        border: 1px solid rgba(50,200,100,0.4);
        border-left: 4px solid #32c864;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
        color: #80ffaa;
        font-size: 0.95rem;
    }

    /* Detection items */
    .detection-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #141418;
        border: 1px solid #222;
        border-radius: 6px;
        padding: 0.7rem 1rem;
        margin: 6px 0;
        font-size: 0.9rem;
    }

    .badge-helmet {
        background: rgba(50,200,100,0.2);
        color: #32c864;
        border: 1px solid rgba(50,200,100,0.4);
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    .badge-head {
        background: rgba(255,107,53,0.2);
        color: #ff6b35;
        border: 1px solid rgba(255,107,53,0.4);
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    .badge-person {
        background: rgba(100,150,255,0.2);
        color: #6496ff;
        border: 1px solid rgba(100,150,255,0.4);
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    .conf-bar-wrap {
        background: #222;
        border-radius: 10px;
        height: 6px;
        width: 100px;
        overflow: hidden;
    }

    .conf-bar-fill {
        height: 100%;
        border-radius: 10px;
        background: linear-gradient(90deg, #ff6b35, #f7c59f);
    }

    /* Section titles */
    .section-title {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.1rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #888;
        border-bottom: 1px solid #222;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0d0d12;
        border-right: 1px solid #1e1e28;
    }

    [data-testid="stSidebar"] .stSlider label {
        color: #888 !important;
        font-size: 0.8rem;
        letter-spacing: 1px;
    }

    /* Upload zone */
    [data-testid="stFileUploader"] {
        background: #0f0f16;
        border: 1px dashed #2a2a3a !important;
        border-radius: 8px;
    }

    /* Image captions */
    .img-caption {
        text-align: center;
        color: #444;
        font-size: 0.75rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-top: 6px;
    }

    /* Status pill */
    .status-pill {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.72rem;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        font-weight: 600;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}

    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #ff6b35, #e55a25) !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        font-family: 'Rajdhani', sans-serif !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        font-weight: 600 !important;
        padding: 0.5rem 2rem !important;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


# ─── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)


# ─── Detection Helper ──────────────────────────────────────────────────────────
def run_detection(model, image_np, conf_thresh, iou_thresh):
    start = time.time()
    results = model(image_np, conf=conf_thresh, iou=iou_thresh, verbose=False)
    inference_ms = (time.time() - start) * 1000

    result_img = results[0].plot()[:, :, ::-1]  # BGR → RGB

    detections = []
    for box in results[0].boxes:
        label = model.names[int(box.cls)]
        confidence = float(box.conf)
        detections.append({"label": label, "confidence": confidence})

    return result_img, detections, inference_ms


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
        <div style='font-size:3rem;'>⛑️</div>
        <div style='font-family:Rajdhani; font-size:1.2rem; letter-spacing:3px;
                    text-transform:uppercase; color:#ff6b35; font-weight:700;'>
            CONTROL PANEL
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    model_path = st.text_input("Model Path", value="models/best.pt",
                                help="Path to your trained best.pt file")

    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
    conf_threshold = st.slider("Confidence Threshold", 0.10, 1.0, 0.35, 0.05,
                                help="Minimum confidence to show a detection")
    iou_threshold = st.slider("IoU Threshold", 0.10, 1.0, 0.50, 0.05,
                               help="Overlap threshold for NMS")

    st.divider()

    st.markdown("""
    <div style='font-size:0.75rem; color:#444; line-height:1.8;'>
        <div style='letter-spacing:2px; text-transform:uppercase; color:#666;
                    margin-bottom:8px; font-family:Rajdhani;'>Model Info</div>
        🧠 Architecture: YOLOv8s<br>
        🏷️ Classes: Head · Helmet · Person<br>
        📐 Input Size: 640×640<br>
        ⚡ Framework: Ultralytics<br>
        🗃️ Dataset: Hard Hat Workers
    </div>
    """, unsafe_allow_html=True)


# ─── Main UI ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">⛑ Helmet Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Safety Compliance System · YOLOv8</div>', unsafe_allow_html=True)

# Load model
if not os.path.exists(model_path):
    st.error(f"❌ Model not found at `{model_path}`. Place `best.pt` in the `models/` folder.")
    st.info("📁 Expected structure:\n```\nhelmet-detection/\n├── models/\n│   └── best.pt  ← your downloaded model\n└── app/\n    └── app.py\n```")
    st.stop()

model = load_model(model_path)
st.markdown(f"""
<div style='text-align:center; margin-bottom:1rem;'>
    <span style='background:rgba(50,200,100,0.15); border:1px solid rgba(50,200,100,0.3);
                 color:#32c864; padding:4px 16px; border-radius:20px; font-size:0.78rem;
                 letter-spacing:2px; text-transform:uppercase;'>
        ● Model Loaded
    </span>
</div>
""", unsafe_allow_html=True)

# Upload Section
st.markdown('<div class="section-title">Upload Image</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Drop an image here or click to browse",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    label_visibility="collapsed"
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.markdown('<div class="section-title">Original Image</div>', unsafe_allow_html=True)
        st.image(image, use_column_width=True)
        st.markdown(f'<div class="img-caption">{image.width} × {image.height} px · {uploaded_file.name}</div>',
                    unsafe_allow_html=True)

    with st.spinner(""):
        result_img, detections, inf_time = run_detection(
            model, image_np, conf_threshold, iou_threshold
        )

    with col2:
        st.markdown('<div class="section-title">Detection Result</div>', unsafe_allow_html=True)
        st.image(result_img, use_column_width=True)
        st.markdown(f'<div class="img-caption">Inference · {inf_time:.1f} ms</div>',
                    unsafe_allow_html=True)

    # Metrics
    st.markdown('<div class="section-title">Analysis Summary</div>', unsafe_allow_html=True)

    total      = len(detections)
    helmets    = sum(1 for d in detections if d["label"].lower() == "helmet")
    heads      = sum(1 for d in detections if d["label"].lower() == "head")
    persons    = sum(1 for d in detections if d["label"].lower() == "person")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total}</div>
            <div class="metric-label">Total Detections</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:#32c864">{helmets}</div>
            <div class="metric-label">Helmets ✅</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:#ff6b35">{heads}</div>
            <div class="metric-label">Heads ⚠️</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:#6496ff">{inf_time:.0f}ms</div>
            <div class="metric-label">Inference Time</div>
        </div>""", unsafe_allow_html=True)

    # Safety Alert
    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
    if heads > 0:
        st.markdown(f"""
        <div class="alert-danger">
            ⚠️ <strong>SAFETY VIOLATION DETECTED</strong> —
            {heads} person(s) found without helmet. Immediate action required.
        </div>""", unsafe_allow_html=True)
    elif helmets > 0:
        st.markdown("""
        <div class="alert-success">
            ✅ <strong>SITE COMPLIANT</strong> —
            All detected persons are wearing helmets. Safety standards met.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background:#141418; border:1px solid #222; border-radius:6px;
                    padding:1rem; color:#555; font-size:0.9rem; margin:1rem 0;'>
            ℹ️ No persons detected in this image. Try lowering the confidence threshold.
        </div>""", unsafe_allow_html=True)

    # Detection Details
    if detections:
        st.markdown('<div class="section-title">Detection Details</div>', unsafe_allow_html=True)
        for i, det in enumerate(detections):
            label = det["label"].lower()
            conf  = det["confidence"]
            conf_pct = int(conf * 100)

            if label == "helmet":
                badge = '<span class="badge-helmet">Helmet</span>'
            elif label == "head":
                badge = '<span class="badge-head">Head</span>'
            else:
                badge = '<span class="badge-person">Person</span>'

            st.markdown(f"""
            <div class="detection-item">
                <span style='color:#888; font-size:0.8rem;'>#{i+1:02d}</span>
                {badge}
                <div style='display:flex; align-items:center; gap:10px;'>
                    <div class='conf-bar-wrap'>
                        <div class='conf-bar-fill' style='width:{conf_pct}%;'></div>
                    </div>
                    <span style='color:#ff6b35; font-family:Rajdhani; font-weight:600;
                                 font-size:1rem;'>{conf:.1%}</span>
                </div>
            </div>""", unsafe_allow_html=True)

    # Download
    st.markdown('<div class="section-title">Export</div>', unsafe_allow_html=True)
    result_pil = Image.fromarray(result_img)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        result_pil.save(tmp.name, quality=95)
        with open(tmp.name, "rb") as f:
            st.download_button(
                "⬇️  Download Result Image",
                data=f,
                file_name=f"helmet_detection_{uploaded_file.name}",
                mime="image/jpeg"
            )

else:
    # Empty state
    st.markdown("""
    <div style='text-align:center; padding: 4rem 2rem; color:#333;'>
        <div style='font-size:4rem; margin-bottom:1rem;'>📸</div>
        <div style='font-family:Rajdhani; font-size:1.3rem; letter-spacing:3px;
                    text-transform:uppercase; color:#444;'>
            Upload an image to begin detection
        </div>
        <div style='font-size:0.8rem; color:#333; margin-top:0.5rem; letter-spacing:1px;'>
            Supported formats: JPG · PNG · BMP · WEBP
        </div>
    </div>
    """, unsafe_allow_html=True)
