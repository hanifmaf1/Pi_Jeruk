import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import os
import time
from pathlib import Path

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Deteksi Kualitas Jeruk",
    page_icon="🍊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Main background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #1a1a2e, #16213e);
    color: #e8e8f0;
}

/* Header */
.hero-header {
    text-align: center;
    padding: 2rem 0 1rem;
}
.hero-title {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #ff6b35, #f7c59f, #ff6b35);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 3s infinite;
    background-size: 200%;
}
@keyframes shimmer {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.hero-subtitle {
    font-size: 1.1rem;
    color: #9aa0b8;
    margin-top: 0.5rem;
    font-weight: 400;
}
.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, #ff6b35, #e63f1e);
    color: white;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-top: 0.75rem;
    letter-spacing: 0.5px;
}

/* Glass card */
.glass-card {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 1.5rem;
    backdrop-filter: blur(10px);
    margin-bottom: 1rem;
}

/* Metric cards */
.metric-row {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    justify-content: center;
    margin: 1rem 0;
}
.metric-card {
    background: rgba(255, 107, 53, 0.12);
    border: 1px solid rgba(255, 107, 53, 0.3);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    text-align: center;
    flex: 1;
    min-width: 130px;
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-3px); }
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #ff6b35;
}
.metric-label {
    font-size: 0.78rem;
    color: #9aa0b8;
    font-weight: 500;
    margin-top: 2px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Class tags */
.class-tag {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 600;
    margin: 3px;
}
.tag-busuk   { background: rgba(231,76,60,0.2);  border: 1px solid #e74c3c; color: #e74c3c; }
.tag-matang-besar  { background: rgba(46,213,115,0.2); border: 1px solid #2ed573; color: #2ed573; }
.tag-matang-sedang { background: rgba(255,165,0,0.2);  border: 1px solid #ffa500; color: #ffa500; }
.tag-mentah  { background: rgba(116,185,255,0.2); border: 1px solid #74b9ff; color: #74b9ff; }

/* Detection result item */
.det-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 0.65rem 1rem;
    margin-bottom: 0.5rem;
    animation: fadeIn 0.4s ease-in;
}
@keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; } }
.det-label { font-weight: 600; font-size: 0.92rem; }
.det-conf  { font-size: 0.82rem; color: #ff6b35; font-weight: 700; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.9) !important;
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* Upload area */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(255, 107, 53, 0.4) !important;
    border-radius: 12px !important;
    background: rgba(255, 107, 53, 0.04) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #ff6b35, #e63f1e);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    font-size: 0.95rem;
    padding: 0.55rem 1.5rem;
    width: 100%;
    transition: opacity 0.2s, transform 0.2s;
}
.stButton > button:hover {
    opacity: 0.9;
    transform: translateY(-1px);
}

/* Progress bar */
.stProgress > div > div { background-color: #ff6b35 !important; }

/* Slider */
.stSlider [data-baseweb="slider"] div[role="slider"]
    { background-color: #ff6b35 !important; }

/* Section divider */
.divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.08);
    margin: 1.5rem 0;
}

/* Info box */
.info-box {
    background: rgba(116,185,255,0.08);
    border-left: 3px solid #74b9ff;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.88rem;
    color: #b0c4de;
}

/* Warning box */
.warn-box {
    background: rgba(255,165,0,0.08);
    border-left: 3px solid #ffa500;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.88rem;
    color: #f0c080;
}
</style>
""", unsafe_allow_html=True)

# ─── Constants ───────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent / "best.pt"

CLASS_INFO = {
    "jeruk busuk": {
        "tag": "tag-busuk",
        "emoji": "🔴",
        "desc": "Jeruk mengalami pembusukan. Tidak layak konsumsi.",
        "color": "#e74c3c",
    },
    "jeruk matang besar": {
        "tag": "tag-matang-besar",
        "emoji": "🟢",
        "desc": "Jeruk matang berukuran besar. Siap dipanen dan dikonsumsi.",
        "color": "#2ed573",
    },
    "jeruk matang sedang": {
        "tag": "tag-matang-sedang",
        "emoji": "🟡",
        "desc": "Jeruk matang berukuran sedang. Siap dipanen dan dikonsumsi.",
        "color": "#ffa500",
    },
    "jeruk mentah": {
        "tag": "tag-mentah",
        "emoji": "🔵",
        "desc": "Jeruk belum matang. Perlu waktu lebih lama untuk panen.",
        "color": "#74b9ff",
    },
}

# ─── Model Loader ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model tidak ditemukan di: {MODEL_PATH}")
        st.stop()
    return YOLO(str(MODEL_PATH))

# ─── Helpers ─────────────────────────────────────────────────────────────────
def run_detection(model, img_array: np.ndarray, conf: float, iou: float):
    results = model(img_array, conf=conf, iou=iou)
    return results

def annotate_image(results) -> np.ndarray:
    return results[0].plot()          # BGR

def parse_detections(results):
    dets = []
    boxes = results[0].boxes
    names = results[0].names
    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls[0])
            label  = names[cls_id]
            conf   = float(box.conf[0])
            dets.append({"label": label, "confidence": conf})
    return dets

def count_classes(dets):
    counts = {}
    for d in dets:
        counts[d["label"]] = counts.get(d["label"], 0) + 1
    return counts

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Pengaturan Deteksi")
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    conf_thresh = st.slider(
        "Ambang Batas Kepercayaan (Confidence)",
        min_value=0.10, max_value=0.99, value=0.25, step=0.01,
        help="Deteksi di bawah nilai ini akan diabaikan",
    )
    iou_thresh = st.slider(
        "Ambang Batas IoU (NMS)",
        min_value=0.1, max_value=0.9, value=0.45, step=0.05,
        help="Threshold Intersection over Union untuk Non-Max Suppression",
    )

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("### 🍊 Kelas Deteksi")
    for cls_name, info in CLASS_INFO.items():
        st.markdown(
            f"<span class='class-tag {info['tag']}'>{info['emoji']} {cls_name.title()}</span>",
            unsafe_allow_html=True,
        )

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("### 📊 Informasi Model")
    st.markdown("""
    <div class='info-box'>
        <b>Model:</b> YOLOv9c<br>
        <b>Dataset:</b> jeruk_final v1 (Roboflow)<br>
        <b>Epochs:</b> 100<br>
        <b>mAP@50:</b> 99.5%<br>
        <b>mAP@50-95:</b> 99.4%
    </div>
    """, unsafe_allow_html=True)

# ─── Hero Header ─────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero-header'>
    <div class='hero-title'>🍊 Deteksi Kualitas Jeruk</div>
    <div class='hero-subtitle'>Sistem klasifikasi kematangan jeruk berbasis YOLOv9 & Computer Vision</div>
    <span class='hero-badge'>YOLOv9c · mAP@50 = 99.5%</span>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Load model (with spinner) ───────────────────────────────────────────────
with st.spinner("⏳ Memuat model YOLOv9…"):
    model = load_model()

# ─── Tab Layout ──────────────────────────────────────────────────────────────
tab_img, tab_cam, tab_info = st.tabs(["📸 Upload Gambar", "📹 Kamera Langsung", "ℹ️ Tentang"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Upload Image
# ══════════════════════════════════════════════════════════════════════════════
with tab_img:
    col_up, col_res = st.columns([1, 1.2], gap="large")

    with col_up:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### 📂 Unggah Gambar")
        uploaded = st.file_uploader(
            "Pilih gambar (JPG / PNG / BMP)",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            label_visibility="collapsed",
        )

        if uploaded:
            pil_img = Image.open(uploaded).convert("RGB")
            st.image(pil_img, caption="Gambar Asli", use_container_width=True)
            detect_btn = st.button("🔍 Deteksi Sekarang", key="detect_img")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_res:
        if uploaded and detect_btn:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("#### 🎯 Hasil Deteksi")

            img_array = np.array(pil_img)
            with st.spinner("Menjalankan inferensi…"):
                t0 = time.perf_counter()
                results = run_detection(model, img_array, conf_thresh, iou_thresh)
                elapsed = (time.perf_counter() - t0) * 1000

            annotated = annotate_image(results)
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, caption="Hasil Anotasi", use_container_width=True)

            dets = parse_detections(results)
            counts = count_classes(dets)

            # ── Metric row ──────────────────────────────────────────────────
            total = len(dets)
            n_mature = counts.get("jeruk matang besar", 0) + counts.get("jeruk matang sedang", 0)
            n_unripe = counts.get("jeruk mentah", 0)
            n_rotten = counts.get("jeruk busuk", 0)

            st.markdown(f"""
            <div class='metric-row'>
                <div class='metric-card'>
                    <div class='metric-value'>{total}</div>
                    <div class='metric-label'>Total Terdeteksi</div>
                </div>
                <div class='metric-card'>
                    <div class='metric-value'>{n_mature}</div>
                    <div class='metric-label'>Matang</div>
                </div>
                <div class='metric-card'>
                    <div class='metric-value'>{n_unripe}</div>
                    <div class='metric-label'>Mentah</div>
                </div>
                <div class='metric-card'>
                    <div class='metric-value'>{n_rotten}</div>
                    <div class='metric-label'>Busuk</div>
                </div>
                <div class='metric-card'>
                    <div class='metric-value'>{elapsed:.0f}ms</div>
                    <div class='metric-label'>Waktu Inferensi</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Per-detection list ───────────────────────────────────────────
            if dets:
                st.markdown("**Detail Setiap Deteksi:**")
                for i, d in enumerate(dets, 1):
                    info = CLASS_INFO.get(d["label"], {})
                    emoji = info.get("emoji", "🍊")
                    color = info.get("color", "#ff6b35")
                    st.markdown(f"""
                    <div class='det-item'>
                        <span class='det-label'>{emoji} #{i} — {d['label'].title()}</span>
                        <span class='det-conf'>{d['confidence']*100:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)

                # ── Class summary bar chart ──────────────────────────────────
                if counts:
                    st.markdown("<br>**Distribusi Kelas:**", unsafe_allow_html=True)
                    chart_data = {k.title(): v for k, v in counts.items()}
                    import pandas as pd
                    df_chart = pd.DataFrame.from_dict(
                        chart_data, orient="index", columns=["Jumlah"]
                    )
                    st.bar_chart(df_chart, color="#ff6b35")
            else:
                st.markdown("""
                <div class='warn-box'>⚠️ Tidak ada objek terdeteksi.
                Coba turunkan <b>Ambang Batas Kepercayaan</b> di panel kiri.</div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

        elif not uploaded:
            st.markdown("""
            <div class='glass-card' style='text-align:center; padding:3rem;'>
                <div style='font-size:3rem;'>🍊</div>
                <div style='color:#9aa0b8; margin-top:0.75rem;'>
                    Unggah gambar jeruk di sebelah kiri untuk memulai deteksi.
                </div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Live Camera
# ══════════════════════════════════════════════════════════════════════════════
with tab_cam:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("#### 📹 Deteksi via Webcam / Kamera")
    st.markdown("""
    <div class='info-box'>
        Fitur ini menggunakan kamera perangkat Anda secara langsung.
        Pastikan browser memiliki izin mengakses kamera.
    </div>
    """, unsafe_allow_html=True)

    camera_img = st.camera_input("Ambil foto dari kamera", label_visibility="collapsed")

    if camera_img:
        pil_cam = Image.open(camera_img).convert("RGB")
        cam_array = np.array(pil_cam)

        col_c1, col_c2 = st.columns(2, gap="medium")
        with col_c1:
            st.image(pil_cam, caption="Gambar Kamera", use_container_width=True)
        with col_c2:
            with st.spinner("Mendeteksi…"):
                t0 = time.perf_counter()
                results_cam = run_detection(model, cam_array, conf_thresh, iou_thresh)
                elapsed_cam = (time.perf_counter() - t0) * 1000

            ann_cam = cv2.cvtColor(annotate_image(results_cam), cv2.COLOR_BGR2RGB)
            st.image(ann_cam, caption="Hasil Deteksi", use_container_width=True)

        dets_cam = parse_detections(results_cam)
        counts_cam = count_classes(dets_cam)
        n_total_cam = len(dets_cam)

        st.markdown(f"""
        <div class='metric-row'>
            <div class='metric-card'>
                <div class='metric-value'>{n_total_cam}</div>
                <div class='metric-label'>Terdeteksi</div>
            </div>
            <div class='metric-card'>
                <div class='metric-value'>{elapsed_cam:.0f}ms</div>
                <div class='metric-label'>Inferensi</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if dets_cam:
            for i, d in enumerate(dets_cam, 1):
                info = CLASS_INFO.get(d["label"], {})
                emoji = info.get("emoji", "🍊")
                st.markdown(f"""
                <div class='det-item'>
                    <span class='det-label'>{emoji} #{i} — {d['label'].title()}</span>
                    <span class='det-conf'>{d['confidence']*100:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='warn-box'>⚠️ Tidak ada jeruk terdeteksi dalam gambar kamera.
            Pastikan jeruk terlihat jelas dan coba turunkan confidence threshold.</div>
            """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — About
# ══════════════════════════════════════════════════════════════════════════════
with tab_info:
    col_a1, col_a2 = st.columns(2, gap="large")

    with col_a1:
        st.markdown("""
        <div class='glass-card'>
            <h4>🔬 Tentang Sistem Ini</h4>
            <p style='color:#9aa0b8; line-height:1.7;'>
            Aplikasi ini menggunakan model <b>YOLOv9c</b> yang telah dilatih khusus
            untuk mendeteksi dan mengklasifikasikan kualitas buah jeruk berdasarkan
            kondisi kematangan dan kesegarannya.
            </p>
            <p style='color:#9aa0b8; line-height:1.7;'>
            Dataset pelatihan diperoleh dari <b>Roboflow</b> (jeruk_final v1)
            yang berisi lebih dari 2.000 gambar jeruk dengan 4 kelas berbeda.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='glass-card'>
            <h4>📈 Performa Model</h4>
            <table style='width:100%; color:#9aa0b8; border-collapse: collapse;'>
                <tr style='border-bottom:1px solid rgba(255,255,255,0.1);'>
                    <td style='padding:8px 0; font-weight:600; color:#e8e8f0;'>Metrik</td>
                    <td style='padding:8px 0; font-weight:600; color:#e8e8f0; text-align:right;'>Nilai</td>
                </tr>
                <tr style='border-bottom:1px solid rgba(255,255,255,0.06);'>
                    <td style='padding:7px 0;'>Precision</td>
                    <td style='padding:7px 0; text-align:right; color:#ff6b35;'>98.5%</td>
                </tr>
                <tr style='border-bottom:1px solid rgba(255,255,255,0.06);'>
                    <td style='padding:7px 0;'>Recall</td>
                    <td style='padding:7px 0; text-align:right; color:#ff6b35;'>100%</td>
                </tr>
                <tr style='border-bottom:1px solid rgba(255,255,255,0.06);'>
                    <td style='padding:7px 0;'>mAP@50</td>
                    <td style='padding:7px 0; text-align:right; color:#2ed573;'>99.5%</td>
                </tr>
                <tr>
                    <td style='padding:7px 0;'>mAP@50-95</td>
                    <td style='padding:7px 0; text-align:right; color:#2ed573;'>99.4%</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with col_a2:
        st.markdown("<div class='glass-card'><h4>🍊 Kelas Deteksi</h4>", unsafe_allow_html=True)
        for cls_name, info in CLASS_INFO.items():
            st.markdown(f"""
            <div style='margin-bottom:1rem; padding:0.75rem 1rem;
                        background:rgba(255,255,255,0.03); border-radius:10px;
                        border-left:3px solid {info["color"]};'>
                <div style='font-weight:600; color:#e8e8f0;'>{info["emoji"]} {cls_name.title()}</div>
                <div style='font-size:0.85rem; color:#9aa0b8; margin-top:3px;'>{info["desc"]}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class='glass-card'>
            <h4>🛠️ Tech Stack</h4>
            <div style='display:flex; flex-wrap:wrap; gap:8px;'>
                <span class='class-tag tag-matang-besar'>YOLOv9</span>
                <span class='class-tag tag-mentah'>Streamlit</span>
                <span class='class-tag tag-matang-sedang'>Ultralytics</span>
                <span class='class-tag tag-busuk'>OpenCV</span>
                <span class='class-tag tag-matang-besar'>PyTorch</span>
                <span class='class-tag tag-mentah'>Roboflow</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding:2rem 0 1rem; color:#555;
            font-size:0.8rem; border-top:1px solid rgba(255,255,255,0.06); margin-top:2rem;'>
    🍊 Sistem Deteksi Kualitas Jeruk · YOLOv9 · Ultralytics · Streamlit
</div>
""", unsafe_allow_html=True)
