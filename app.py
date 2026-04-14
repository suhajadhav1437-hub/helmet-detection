import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import os

# Start your app logic here
model = YOLO('best.pt')

# Rest of your app logic...
st.set_page_config(page_title="Helmet Detection System", page_icon="⛑️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@600;700&family=Inter:wght@300;400;500&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:#0a0a0f;color:#e0e0e0;}
.main-header{font-family:'Rajdhani',sans-serif;font-size:2.8rem;font-weight:700;letter-spacing:4px;
    text-transform:uppercase;background:linear-gradient(135deg,#ff6b35,#f7c59f);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;text-align:center;padding:1.2rem 0 0.2rem 0;}
.sub-header{text-align:center;color:#555;font-size:0.82rem;letter-spacing:3px;text-transform:uppercase;margin-bottom:1.5rem;}
.metric-card{background:linear-gradient(135deg,#141418,#1a1a22);border:1px solid #2a2a35;
    border-radius:8px;padding:1rem;text-align:center;position:relative;overflow:hidden;}
.metric-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;
    background:linear-gradient(90deg,#ff6b35,#f7c59f);}
.metric-value{font-family:'Rajdhani',sans-serif;font-size:2rem;font-weight:700;color:#ff6b35;line-height:1;}
.metric-label{font-size:0.68rem;letter-spacing:2px;text-transform:uppercase;color:#555;margin-top:4px;}
.alert-danger{background:rgba(220,50,50,0.12);border:1px solid rgba(220,50,50,0.35);
    border-left:4px solid #dc3232;border-radius:6px;padding:0.9rem 1.2rem;margin:1rem 0;color:#ff8080;}
.alert-success{background:rgba(50,200,100,0.12);border:1px solid rgba(50,200,100,0.35);
    border-left:4px solid #32c864;border-radius:6px;padding:0.9rem 1.2rem;margin:1rem 0;color:#80ffaa;}
.section-title{font-family:'Rajdhani',sans-serif;font-size:1rem;letter-spacing:3px;text-transform:uppercase;
    color:#666;border-bottom:1px solid #1e1e28;padding-bottom:0.4rem;margin:1.2rem 0 0.8rem 0;}
.det-item{display:flex;align-items:center;justify-content:space-between;background:#141418;
    border:1px solid #1e1e28;border-radius:6px;padding:0.6rem 1rem;margin:5px 0;font-size:0.88rem;}
.badge{padding:2px 10px;border-radius:20px;font-size:0.72rem;font-weight:600;letter-spacing:1px;text-transform:uppercase;}
.badge-helmet{background:rgba(50,200,100,0.15);color:#32c864;border:1px solid rgba(50,200,100,0.35);}
.badge-head{background:rgba(255,107,53,0.15);color:#ff6b35;border:1px solid rgba(255,107,53,0.35);}
.badge-person{background:rgba(100,150,255,0.15);color:#6496ff;border:1px solid rgba(100,150,255,0.35);}
[data-testid="stSidebar"]{background:#0d0d12;border-right:1px solid #1e1e28;}
#MainMenu{visibility:hidden;}footer{visibility:hidden;}.stDeployButton{display:none;}
.stDownloadButton>button{background:linear-gradient(135deg,#ff6b35,#e55a25)!important;
    color:white!important;border:none!important;border-radius:6px!important;
    font-family:'Rajdhani',sans-serif!important;letter-spacing:2px!important;
    text-transform:uppercase!important;font-weight:600!important;width:100%;}
</style>
""", unsafe_allow_html=True)


def draw_boxes_pil(image_pil, results, model):
    draw   = ImageDraw.Draw(image_pil)
    colors = {"helmet":"#32c864","head":"#ff6b35","person":"#6496ff"}
    detections = []
    for box in results[0].boxes:
        x1,y1,x2,y2 = [int(v) for v in box.xyxy[0].tolist()]
        label      = model.names[int(box.cls)]
        confidence = float(box.conf)
        color      = colors.get(label.lower(),"#ffffff")
        for t in range(3):
            draw.rectangle([x1-t,y1-t,x2+t,y2+t], outline=color)
        text = f"{label} {confidence:.0%}"
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",13)
        except Exception:
            font = ImageFont.load_default()
        tw = len(text)*8
        draw.rectangle([x1,y1-22,x1+tw+8,y1], fill=color)
        draw.text((x1+4,y1-20), text, fill="white", font=font)
        detections.append({"label":label,"confidence":confidence})
    return image_pil, detections


@st.cache_resource
def load_model(path):
    return YOLO(path)


with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0;'>
        <div style='font-size:2.8rem;'>⛑️</div>
        <div style='font-family:Rajdhani;font-size:1.1rem;letter-spacing:3px;
                    text-transform:uppercase;color:#ff6b35;font-weight:700;'>CONTROL PANEL</div>
    </div>""", unsafe_allow_html=True)
    st.divider()
    model_path     = st.text_input("Model Path", value="best.pt")
    conf_threshold = st.slider("Confidence Threshold", 0.10, 1.0, 0.35, 0.05)
    iou_threshold  = st.slider("IoU Threshold",         0.10, 1.0, 0.50, 0.05)
    st.divider()
    st.markdown("""
    <div style='font-size:0.72rem;color:#444;line-height:1.9;'>
        <div style='letter-spacing:2px;text-transform:uppercase;color:#555;
                    margin-bottom:6px;font-family:Rajdhani;'>Model Info</div>
        🧠 YOLOv8s &nbsp;|&nbsp; ⚡ Ultralytics<br>
        🏷️ Head · Helmet · Person<br>
        📐 640×640 input
    </div>""", unsafe_allow_html=True)

st.markdown('<div class="main-header">⛑ Helmet Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Safety Compliance · YOLOv8</div>', unsafe_allow_html=True)

if not os.path.exists(model_path):
    st.error(f"❌ Model not found at `{model_path}`. Make sure `best.pt` is uploaded to the repository root.")
    st.stop()

model = load_model(model_path)
st.markdown("""
<div style='text-align:center;margin-bottom:1rem;'>
    <span style='background:rgba(50,200,100,0.12);border:1px solid rgba(50,200,100,0.3);
                 color:#32c864;padding:3px 14px;border-radius:20px;font-size:0.76rem;
                 letter-spacing:2px;text-transform:uppercase;'>● Model Loaded</span>
</div>""", unsafe_allow_html=True)

st.markdown('<div class="section-title">Upload Image</div>', unsafe_allow_html=True)
uploaded = st.file_uploader("Drop an image or browse", type=["jpg","jpeg","png","bmp","webp"], label_visibility="collapsed")

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.markdown('<div class="section-title">Original Image</div>', unsafe_allow_html=True)
        st.image(image, use_column_width=True)
        st.markdown(f'<div style="text-align:center;color:#444;font-size:0.72rem;letter-spacing:2px;text-transform:uppercase;">{image.width}×{image.height} px</div>', unsafe_allow_html=True)

    with st.spinner("🔍 Detecting..."):
        start   = time.time()
        results = model(np.array(image), conf=conf_threshold, iou=iou_threshold, verbose=False)
        inf_ms  = (time.time()-start)*1000
        result_img, detections = draw_boxes_pil(image.copy(), results, model)

    with col2:
        st.markdown('<div class="section-title">Detection Result</div>', unsafe_allow_html=True)
        st.image(result_img, use_column_width=True)
        st.markdown(f'<div style="text-align:center;color:#444;font-size:0.72rem;letter-spacing:2px;text-transform:uppercase;">Inference · {inf_ms:.1f} ms</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Analysis Summary</div>', unsafe_allow_html=True)
    total   = len(detections)
    helmets = sum(1 for d in detections if d["label"].lower()=="helmet")
    heads   = sum(1 for d in detections if d["label"].lower()=="head")

    m1,m2,m3,m4 = st.columns(4)
    with m1: st.markdown(f'<div class="metric-card"><div class="metric-value">{total}</div><div class="metric-label">Total</div></div>', unsafe_allow_html=True)
    with m2: st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#32c864">{helmets}</div><div class="metric-label">Helmets ✅</div></div>', unsafe_allow_html=True)
    with m3: st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#ff6b35">{heads}</div><div class="metric-label">Heads ⚠️</div></div>', unsafe_allow_html=True)
    with m4: st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#6496ff">{inf_ms:.0f}ms</div><div class="metric-label">Inference</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
    if heads > 0:
        st.markdown(f'<div class="alert-danger">⚠️ <strong>SAFETY VIOLATION</strong> — {heads} person(s) without helmet!</div>', unsafe_allow_html=True)
    elif helmets > 0:
        st.markdown('<div class="alert-success">✅ <strong>SITE COMPLIANT</strong> — All persons wearing helmets.</div>', unsafe_allow_html=True)

    if detections:
        st.markdown('<div class="section-title">Detection Details</div>', unsafe_allow_html=True)
        for i,det in enumerate(detections):
            lbl  = det["label"].lower()
            conf = det["confidence"]
            badge_class = lbl if lbl in ("helmet","head","person") else "person"
            bar_w = int(conf*100)
            st.markdown(f"""
            <div class="det-item">
                <span style='color:#555;font-size:0.78rem;'>#{i+1:02d}</span>
                <span class="badge badge-{badge_class}">{det["label"]}</span>
                <div style='display:flex;align-items:center;gap:10px;'>
                    <div style='background:#222;border-radius:10px;height:6px;width:90px;overflow:hidden;'>
                        <div style='width:{bar_w}%;height:100%;border-radius:10px;background:linear-gradient(90deg,#ff6b35,#f7c59f);'></div>
                    </div>
                    <span style='color:#ff6b35;font-family:Rajdhani;font-weight:600;font-size:0.95rem;'>{conf:.1%}</span>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Export</div>', unsafe_allow_html=True)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        result_img.save(tmp.name, format="PNG")
        with open(tmp.name,"rb") as f:
            st.download_button("⬇️  Download Result", data=f,
                file_name=f"helmet_result_{uploaded.name}", mime="image/png")
else:
    st.markdown("""
    <div style='text-align:center;padding:3.5rem 2rem;color:#2a2a35;'>
        <div style='font-size:3.5rem;margin-bottom:1rem;'>📸</div>
        <div style='font-family:Rajdhani;font-size:1.2rem;letter-spacing:3px;text-transform:uppercase;color:#333;'>Upload an image to begin</div>
        <div style='font-size:0.78rem;color:#2a2a35;margin-top:0.5rem;letter-spacing:1px;'>JPG · PNG · BMP · WEBP</div>
    </div>""", unsafe_allow_html=True)
