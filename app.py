

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io
import base64
import time
 
# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="FruitSense AI – Fresh vs Rotten",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
# ----------------- CUSTOM CSS -----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&display=swap');
 
/* ===== ROOT THEME ===== */
:root {
    --bg-deep:     #0d1117;
    --bg-card:     #161b22;
    --bg-card2:    #1c2333;
    --accent-green:#39d353;
    --accent-red:  #f85149;
    --accent-gold: #e3b341;
    --text-primary:#e6edf3;
    --text-muted:  #8b949e;
    --border:      rgba(255,255,255,0.07);
    --glow-green:  0 0 30px rgba(57,211,83,0.25);
    --glow-red:    0 0 30px rgba(248,81,73,0.25);
    --radius:      18px;
}
 
/* ===== GLOBAL ===== */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-deep) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--text-primary);
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
 
/* ===== HERO ===== */
.hero {
    text-align: center;
    padding: 48px 0 20px;
    position: relative;
}
.hero-badge {
    display: inline-block;
    background: rgba(57,211,83,0.1);
    border: 1px solid rgba(57,211,83,0.3);
    color: var(--accent-green);
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 5px 16px;
    border-radius: 100px;
    margin-bottom: 20px;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.4rem, 5vw, 4rem);
    font-weight: 900;
    line-height: 1.1;
    background: linear-gradient(135deg, #e6edf3 0%, #39d353 60%, #e3b341 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 12px;
}
.hero-sub {
    color: var(--text-muted);
    font-size: 1.05rem;
    font-weight: 300;
    max-width: 480px;
    margin: 0 auto;
}
 
/* ===== UPLOAD ZONE ===== */
.upload-zone {
    background: var(--bg-card);
    border: 2px dashed rgba(57,211,83,0.3);
    border-radius: var(--radius);
    padding: 36px 24px;
    text-align: center;
    transition: border-color 0.3s;
    margin: 24px 0;
}
.upload-zone:hover { border-color: var(--accent-green); }
.upload-icon { font-size: 3rem; margin-bottom: 12px; }
.upload-hint { color: var(--text-muted); font-size: 0.9rem; }
 
/* ===== PREVIEW CARD ===== */
.preview-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.preview-header {
    padding: 14px 20px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.85rem;
    color: var(--text-muted);
    font-weight: 500;
}
.dot { width:10px; height:10px; border-radius:50%; display:inline-block; }
.dot-red{background:#f85149;} .dot-yellow{background:#e3b341;} .dot-green{background:#39d353;}
 
/* ===== RESULT CARD ===== */
.result-wrapper {
    background: var(--bg-card);
    border-radius: var(--radius);
    padding: 32px 28px;
    border: 1px solid var(--border);
    position: relative;
    overflow: hidden;
    margin-top: 20px;
}
.result-wrapper::before {
    content: '';
    position: absolute; top:0; left:0; right:0; height:3px;
    background: var(--bar-color, var(--accent-green));
}
.result-label {
    font-family: 'Playfair Display', serif;
    font-size: 2.4rem;
    font-weight: 700;
    margin: 0 0 6px;
}
.result-label.fresh { color: var(--accent-green); text-shadow: var(--glow-green); }
.result-label.rotten { color: var(--accent-red); text-shadow: var(--glow-red); }
.result-confidence {
    font-size: 1rem;
    color: var(--text-muted);
    margin-bottom: 20px;
}
 
/* ===== CONFIDENCE BAR ===== */
.conf-track {
    background: rgba(255,255,255,0.07);
    border-radius: 100px;
    height: 8px;
    overflow: hidden;
    margin: 4px 0 20px;
}
.conf-fill {
    height: 100%;
    border-radius: 100px;
    background: var(--fill-color, var(--accent-green));
    transition: width 1s ease;
}
 
/* ===== SUGGESTION BOX ===== */
.suggestion {
    background: var(--bg-card2);
    border-left: 3px solid var(--sug-color, var(--accent-green));
    border-radius: 8px;
    padding: 14px 18px;
    font-size: 0.92rem;
    color: var(--text-primary);
    margin-top: 16px;
    line-height: 1.6;
}
.suggestion strong { color: var(--sug-color, var(--accent-green)); }
 
/* ===== METRICS ROW ===== */
.metrics-row {
    display: flex;
    gap: 12px;
    margin: 20px 0 0;
    flex-wrap: wrap;
}
.metric-chip {
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 10px 16px;
    flex: 1;
    min-width: 100px;
    text-align: center;
}
.metric-chip .m-label { font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; }
.metric-chip .m-value { font-size: 1.15rem; font-weight: 600; margin-top: 3px; }
 
/* ===== BATCH GRID ===== */
.batch-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
    gap: 12px;
    margin-top: 20px;
}
.batch-item {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
    text-align: center;
    padding-bottom: 10px;
}
.batch-item img { width:100%; aspect-ratio:1; object-fit:cover; }
.batch-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 100px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-top: 6px;
}
.badge-fresh { background: rgba(57,211,83,0.15); color: var(--accent-green); }
.badge-rotten { background: rgba(248,81,73,0.15); color: var(--accent-red); }
 
/* ===== PRIVACY PILL ===== */
.privacy-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(227,179,65,0.1);
    border: 1px solid rgba(227,179,65,0.25);
    color: var(--accent-gold);
    font-size: 0.78rem;
    padding: 4px 14px;
    border-radius: 100px;
    margin: 16px 0 0;
}
 
/* ===== DIVIDER ===== */
.section-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 32px 0;
}
 
/* ===== STREAMLIT OVERRIDES ===== */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border-radius: var(--radius) !important;
    border: 2px dashed rgba(57,211,83,0.3) !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(57,211,83,0.7) !important;
}
.stButton > button {
    background: linear-gradient(135deg, #39d353, #2ea043) !important;
    color: #0d1117 !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 12px 32px !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
    box-shadow: 0 4px 20px rgba(57,211,83,0.3) !important;
}
.stButton > button:hover { opacity: 0.88 !important; }
.stButton > button:active { opacity: 0.75 !important; }
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}
.stProgress > div > div {
    background: linear-gradient(90deg, #39d353, #2ea043) !important;
    border-radius: 100px !important;
}
</style>
""", unsafe_allow_html=True)
 
 
# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_classifier():
    return load_model("fruit_fresh_rotten_classifier.keras")
 
def img_to_b64(pil_img: Image.Image, size=(200, 200)) -> str:
    pil_img = pil_img.copy()
    pil_img.thumbnail(size)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()
 
def predict_single(model, pil_img: Image.Image, h: int, w: int):
    img = pil_img.resize((w, h))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    raw = float(model.predict(arr, verbose=0)[0][0])
    if raw > 0.5:
        return "Rotten", raw * 100, raw
    else:
        return "Fresh", (1 - raw) * 100, raw
 
def confidence_tier(conf: float) -> str:
    if conf >= 85:   return "High"
    if conf >= 60:   return "Medium"
    return "Low"
 
def suggestion_text(label: str, conf: float) -> str:
    tier = confidence_tier(conf)
    if label == "Fresh":
        if tier == "High":
            return "<strong>✅ Safe to eat!</strong> This fruit looks fresh and healthy. Enjoy it as-is, add to a smoothie, or use in a recipe."
        elif tier == "Medium":
            return "<strong>🟡 Probably fine.</strong> The fruit appears mostly fresh but inspect it closely — check for any soft spots before eating."
        else:
            return "<strong>⚠️ Uncertain.</strong> The image quality may be low. Please retake with better lighting for a more reliable result."
    else:
        if tier == "High":
            return "<strong>❌ Do not eat.</strong> Clear signs of spoilage detected. Consider composting this fruit to avoid food waste guilt."
        elif tier == "Medium":
            return "<strong>🟠 Likely spoiled.</strong> Some decay indicators found. When in doubt, throw it out — food safety first!"
        else:
            return "<strong>⚠️ Unclear result.</strong> Please retake the photo in bright, natural light with the fruit centered in the frame."
 
def validate_image(pil_img: Image.Image, file_size_mb: float) -> tuple[bool, str]:
    if file_size_mb > 10:
        return False, f"File is {file_size_mb:.1f} MB — please use an image under 10 MB."
    w, h = pil_img.size
    if w < 50 or h < 50:
        return False, "Image is too small. Please upload a clearer photo."
    return True, ""
 
 
# ─────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────
try:
    model = load_classifier()
    input_shape = model.input_shape
    IMG_H, IMG_W = input_shape[1], input_shape[2]
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_err = str(e)
 
 
# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🍃 FruitSense AI")
    st.markdown("---")
    st.markdown("**Model Info**")
    if model_loaded:
        st.success(f"Model loaded ✓")
        st.markdown(f"""
| Property | Value |
|---|---|
| Input size | `{IMG_H} × {IMG_W}` |
| Type | Binary CNN |
| Classes | Fresh / Rotten |
""")
    else:
        st.error("Model not loaded")
 
    st.markdown("---")
    st.markdown("**Mode**")
    app_mode = st.radio("Select mode", ["Single Image", "Batch Upload"], label_visibility="collapsed")
 
    st.markdown("---")
    st.markdown("**Settings**")
    show_raw = st.toggle("Show raw prediction value", value=False)
    show_tips = st.toggle("Show actionable suggestions", value=True)
 
    st.markdown("---")
    st.markdown("""
<div class='privacy-pill'>🔒 No images stored or transmitted</div>
<p style='font-size:0.75rem;color:#8b949e;margin-top:10px;line-height:1.5;'>
All inference runs locally. Your photos never leave your device.
</p>
""", unsafe_allow_html=True)
 
 
# ─────────────────────────────────────────────
#  HERO HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class='hero'>
    <div class='hero-badge'>🧠 AI-Powered Freshness Detection</div>
    <h1 class='hero-title'>FruitSense AI</h1>
    <p class='hero-sub'>Instantly detect fresh or rotten fruit using computer vision — fast, private, and reliable.</p>
</div>
""", unsafe_allow_html=True)
 
if not model_loaded:
    st.error(f"⚠️ Could not load model: `{model_err}`\n\nMake sure `fruit_fresh_rotten_classifier.keras` is in the same directory.")
    st.stop()
 
 
# ─────────────────────────────────────────────
#  SINGLE IMAGE MODE
# ─────────────────────────────────────────────
if app_mode == "Single Image":
 
    col_upload, col_result = st.columns([1, 1], gap="large")
 
    with col_upload:
        st.markdown("#### 📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Drag & drop or browse",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed"
        )
        st.markdown("""
<p class='upload-hint' style='text-align:center;margin-top:8px;'>
🖼️ Supported: JPG, PNG, WEBP &nbsp;|&nbsp; Max: 10 MB<br>
💡 Tip: Center the fruit in the frame for best results
</p>
""", unsafe_allow_html=True)
 
        if uploaded_file:
            file_mb = uploaded_file.size / 1e6
            image = Image.open(uploaded_file).convert("RGB")
            valid, err_msg = validate_image(image, file_mb)
 
            if not valid:
                st.error(f"🚫 {err_msg}")
            else:
                # Preview
                st.markdown("""
<div class='preview-card'>
  <div class='preview-header'>
    <span class='dot dot-red'></span>
    <span class='dot dot-yellow'></span>
    <span class='dot dot-green'></span>
    &nbsp; Image Preview
  </div>
</div>
""", unsafe_allow_html=True)
                st.image(image, use_container_width=True)
                st.caption(f"📁 {uploaded_file.name} &nbsp;|&nbsp; {file_mb:.2f} MB &nbsp;|&nbsp; {image.size[0]}×{image.size[1]}px")
 
                predict_btn = st.button("🔍 Analyze Freshness", use_container_width=True)
 
    with col_result:
        st.markdown("#### 📊 Analysis Result")
 
        if uploaded_file and valid:
            if predict_btn:
                with st.spinner("Analyzing…"):
                    start = time.time()
                    label, confidence, raw_val = predict_single(model, image, IMG_H, IMG_W)
                    elapsed = (time.time() - start) * 1000  # ms
 
                is_fresh = label == "Fresh"
                bar_color = "#39d353" if is_fresh else "#f85149"
                css_class = "fresh" if is_fresh else "rotten"
                tier = confidence_tier(confidence)
 
                tier_color = {
                    "High": "#39d353",
                    "Medium": "#e3b341",
                    "Low": "#f85149"
                }[tier]
 
                st.markdown(f"""
<div class='result-wrapper' style='--bar-color:{bar_color}'>
  <div class='result-label {css_class}'>{label}</div>
  <div class='result-confidence'>Confidence — <strong style='color:{tier_color}'>{tier}</strong> ({confidence:.1f}%)</div>
  <div class='conf-track'>
    <div class='conf-fill' style='width:{confidence:.0f}%;--fill-color:{bar_color}'></div>
  </div>
  <div class='metrics-row'>
    <div class='metric-chip'>
        <div class='m-label'>Score</div>
        <div class='m-value' style='color:{bar_color}'>{confidence:.1f}%</div>
    </div>
    <div class='metric-chip'>
        <div class='m-label'>Certainty</div>
        <div class='m-value' style='color:{tier_color}'>{tier}</div>
    </div>
    <div class='metric-chip'>
        <div class='m-label'>Speed</div>
        <div class='m-value'>{elapsed:.0f}ms</div>
    </div>
  </div>
  {"<div class='suggestion' style='--sug-color:" + bar_color + "'>" + suggestion_text(label, confidence) + "</div>" if show_tips else ""}
</div>
""", unsafe_allow_html=True)
 
                # Raw debug
                if show_raw:
                    with st.expander("🔬 Raw Prediction Value"):
                        st.code(f"raw output: {raw_val:.6f}\n(>0.5 = Rotten, ≤0.5 = Fresh)", language="text")
 
                # Action buttons
                st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1:
                    # Download result as text
                    result_txt = f"FruitSense AI Result\n\nFile: {uploaded_file.name}\nLabel: {label}\nConfidence: {confidence:.2f}%\nCertainty: {tier}\nInference: {elapsed:.0f}ms"
                    st.download_button("💾 Save Result", data=result_txt, file_name="fruitsense_result.txt", mime="text/plain", use_container_width=True)
                with c2:
                    if st.button("🔄 Retake / New Image", use_container_width=True):
                        st.rerun()
 
        else:
            # Empty state
            st.markdown("""
<div style='background:var(--bg-card);border:1px solid var(--border);border-radius:18px;
padding:60px 24px;text-align:center;color:var(--text-muted);'>
    <div style='font-size:3rem;margin-bottom:16px;'>🔬</div>
    <div style='font-size:1rem;font-weight:500;color:#e6edf3;'>Awaiting image…</div>
    <div style='font-size:0.85rem;margin-top:8px;'>Upload a fruit photo on the left to begin analysis</div>
</div>
""", unsafe_allow_html=True)
 
 
# ─────────────────────────────────────────────
#  BATCH MODE
# ─────────────────────────────────────────────
else:
    st.markdown("#### 📦 Batch Upload — Classify Multiple Fruits")
    st.caption("Upload up to 10 images at once. Each will be analyzed individually.")
 
    uploaded_files = st.file_uploader(
        "Select multiple images",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
 
    if uploaded_files:
        if len(uploaded_files) > 10:
            st.warning("⚠️ Maximum 10 images allowed. Only the first 10 will be processed.")
            uploaded_files = uploaded_files[:10]
 
        if st.button(f"🔍 Classify All ({len(uploaded_files)} images)", use_container_width=True):
            progress_bar = st.progress(0, text="Classifying…")
            results = []
 
            for i, f in enumerate(uploaded_files):
                img = Image.open(f).convert("RGB")
                file_mb = f.size / 1e6
                valid, err = validate_image(img, file_mb)
                if not valid:
                    results.append({"name": f.name, "image": img, "error": err})
                else:
                    label, confidence, raw = predict_single(model, img, IMG_H, IMG_W)
                    results.append({"name": f.name, "image": img, "label": label, "confidence": confidence})
                progress_bar.progress((i + 1) / len(uploaded_files), text=f"Processing {i+1}/{len(uploaded_files)}…")
 
            progress_bar.empty()
 
            # Summary stats
            fresh_count = sum(1 for r in results if r.get("label") == "Fresh")
            rotten_count = sum(1 for r in results if r.get("label") == "Rotten")
            err_count = sum(1 for r in results if "error" in r)
 
            st.markdown(f"""
<div style='display:flex;gap:12px;margin:20px 0;flex-wrap:wrap;'>
  <div class='metric-chip' style='background:rgba(57,211,83,0.1);border-color:rgba(57,211,83,0.3);'>
    <div class='m-label'>Fresh</div>
    <div class='m-value' style='color:#39d353;'>{fresh_count}</div>
  </div>
  <div class='metric-chip' style='background:rgba(248,81,73,0.1);border-color:rgba(248,81,73,0.3);'>
    <div class='m-label'>Rotten</div>
    <div class='m-value' style='color:#f85149;'>{rotten_count}</div>
  </div>
  <div class='metric-chip'>
    <div class='m-label'>Total</div>
    <div class='m-value'>{len(results)}</div>
  </div>
</div>
""", unsafe_allow_html=True)
 
            # Results grid
            cols = st.columns(min(len(results), 4))
            for idx, r in enumerate(results):
                with cols[idx % 4]:
                    st.image(r["image"], use_container_width=True)
                    if "error" in r:
                        st.markdown(f"<div class='batch-badge' style='background:rgba(227,179,65,0.15);color:#e3b341;'>⚠️ Error</div>", unsafe_allow_html=True)
                        st.caption(r["error"])
                    else:
                        badge_cls = "badge-fresh" if r["label"] == "Fresh" else "badge-rotten"
                        icon = "✅" if r["label"] == "Fresh" else "❌"
                        st.markdown(f"<div class='batch-badge {badge_cls}'>{icon} {r['label']} — {r['confidence']:.0f}%</div>", unsafe_allow_html=True)
                    st.caption(r["name"][:22] + "…" if len(r["name"]) > 22 else r["name"])
 
    else:
        st.markdown("""
<div style='background:var(--bg-card);border:2px dashed rgba(57,211,83,0.2);border-radius:18px;
padding:60px 24px;text-align:center;color:var(--text-muted);margin-top:16px;'>
    <div style='font-size:3rem;margin-bottom:16px;'>📦</div>
    <div style='font-size:1rem;font-weight:500;color:#e6edf3;'>No images selected</div>
    <div style='font-size:0.85rem;margin-top:8px;'>Click above to select multiple fruit images</div>
</div>
""", unsafe_allow_html=True)
 
 
# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align:center;color:#8b949e;font-size:0.8rem;'>
🍃 FruitSense AI &nbsp;·&nbsp; For guidance only — always inspect food manually &nbsp;·&nbsp;
🔒 Privacy-first: no image data stored or transmitted &nbsp;·&nbsp;
👨‍💻 Developed by <b style='color:#4CAF50;'>Hardik Sarvaiya</b>
</p>
""", unsafe_allow_html=True)
