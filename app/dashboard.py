import streamlit as st
import requests
import time
from datetime import datetime

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="DefectScan AI",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────── CSS ───────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;600;700;800&display=swap');

:root {
    --red:    #ff4422;
    --green:  #00ffaa;
    --blue:   #4488ff;
    --bg:     #07080f;
    --panel:  rgba(255,255,255,0.03);
    --border: rgba(255,255,255,0.07);
    --mono:   'Space Mono', monospace;
    --sans:   'Syne', sans-serif;
}

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: #dde0ee;
    font-family: var(--sans);
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stAppViewContainer"] { background: var(--bg) !important; }

/* Noise grain overlay */
body::after {
    content:'';
    position:fixed; inset:0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.04'/%3E%3C/svg%3E");
    pointer-events:none; z-index:9998; opacity:0.35;
}

/* Scanline overlay */
body::before {
    content:'';
    position:fixed; inset:0;
    background: repeating-linear-gradient(
        0deg, transparent, transparent 3px,
        rgba(255,255,255,0.013) 3px, rgba(255,255,255,0.013) 4px
    );
    pointer-events:none; z-index:9997;
}

/* HEADER */
.hdr {
    display:flex; align-items:flex-start; justify-content:space-between;
    padding: 2.2rem 0 1.6rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2.4rem;
    position: relative; z-index:10;
}
.brand-eye {
    font-family: var(--mono);
    font-size: 9px; letter-spacing:5px;
    color: var(--red); text-transform:uppercase;
    margin-bottom:.4rem;
    animation: fadeSlideDown 0.6s ease both;
}
.brand-name {
    font-size:54px; font-weight:800; line-height:1;
    letter-spacing:-3px; color:#eeeeff;
    animation: fadeSlideDown 0.7s ease both;
}
.brand-name em { color:var(--red); font-style:normal; }
.brand-sub {
    font-family:var(--mono); font-size:10px;
    color:rgba(255,255,255,0.25); letter-spacing:1.5px;
    margin-top:.5rem;
    animation: fadeSlideDown 0.8s ease both;
}
@keyframes fadeSlideDown {
    from { opacity:0; transform:translateY(-12px); }
    to   { opacity:1; transform:translateY(0); }
}

.live-badge {
    display:inline-flex; align-items:center; gap:8px;
    background:rgba(255,255,255,0.04);
    border:1px solid rgba(255,255,255,0.08);
    border-radius:100px; padding:8px 16px;
    font-family:var(--mono); font-size:10px;
    color:rgba(255,255,255,0.45);
    animation: fadeSlideDown 0.9s ease both;
}
.live-dot {
    width:7px; height:7px; border-radius:50%;
    background:var(--green); box-shadow:0 0 10px var(--green);
    animation: livepulse 1.8s ease-in-out infinite;
}
@keyframes livepulse {
    0%,100%{ transform:scale(1); opacity:1; }
    50%    { transform:scale(0.5); opacity:0.4; }
}

/* UPLOAD ZONE */
[data-testid="stFileUploader"] > div {
    background: rgba(255,255,255,0.02) !important;
    border: 1.5px dashed rgba(255,255,255,0.09) !important;
    border-radius: 14px !important;
    padding: 2rem !important;
    transition: border-color .3s, background .3s;
    position: relative; z-index:10;
}
[data-testid="stFileUploader"] > div:hover {
    border-color: rgba(255,68,34,0.35) !important;
    background: rgba(255,68,34,0.03) !important;
}
[data-testid="stFileUploader"] label {
    font-family:var(--mono) !important; font-size:11px !important;
    color:rgba(255,255,255,0.35) !important; letter-spacing:2px !important;
}

/* IMAGE FRAME */
.img-frame {
    position:relative; border-radius:12px; overflow:hidden;
    border:1px solid rgba(255,255,255,0.09);
    z-index:10;
}
.img-frame::before {
    content:''; position:absolute; top:0; left:0; right:0; height:2px;
    background: linear-gradient(90deg, var(--red), #ff9070, transparent);
    z-index:20;
}

/* Scanning sweep line */
.sweep {
    position:absolute; left:0; right:0; height:3px; z-index:25;
    background: linear-gradient(90deg,
        transparent 0%, rgba(0,255,170,0.0) 10%,
        rgba(0,255,170,0.8) 50%,
        rgba(0,255,170,0.0) 90%, transparent 100%
    );
    box-shadow: 0 0 18px 4px rgba(0,255,170,0.35);
    animation: sweep-scan 2.8s ease-in-out infinite;
}
@keyframes sweep-scan {
    0%   { top:0%;    opacity:0; }
    5%   { opacity:1; }
    95%  { opacity:1; }
    100% { top:100%;  opacity:0; }
}

/* Corner brackets */
.c { position:absolute; width:18px; height:18px; z-index:30; }
.c-tl { top:9px; left:9px;   border-top:2px solid var(--red); border-left:2px solid var(--red);
        animation: corner-pulse 2.5s ease-in-out infinite; }
.c-tr { top:9px; right:9px;  border-top:2px solid var(--red); border-right:2px solid var(--red);
        animation: corner-pulse 2.5s 0.3s ease-in-out infinite; }
.c-bl { bottom:9px; left:9px;  border-bottom:2px solid var(--red); border-left:2px solid var(--red);
        animation: corner-pulse 2.5s 0.6s ease-in-out infinite; }
.c-br { bottom:9px; right:9px; border-bottom:2px solid var(--red); border-right:2px solid var(--red);
        animation: corner-pulse 2.5s 0.9s ease-in-out infinite; }
@keyframes corner-pulse {
    0%,100%{ opacity:1; } 50%{ opacity:0.3; }
}

/* Crosshair reticle */
.reticle {
    position:absolute; top:50%; left:50%;
    transform:translate(-50%,-50%);
    width:42px; height:42px; z-index:30;
    border:1px solid rgba(0,255,170,0.22);
    border-radius:50%;
    animation: reticle-spin 7s linear infinite;
}
.reticle::before, .reticle::after {
    content:''; position:absolute; background:rgba(0,255,170,0.28);
}
.reticle::before { left:50%; top:0; width:1px; height:100%; transform:translateX(-50%); }
.reticle::after  { top:50%; left:0; height:1px; width:100%; transform:translateY(-50%); }
@keyframes reticle-spin {
    from { transform:translate(-50%,-50%) rotate(0deg); }
    to   { transform:translate(-50%,-50%) rotate(360deg); }
}

.img-label {
    position:absolute; bottom:10px; left:10px;
    font-family:var(--mono); font-size:9px; letter-spacing:2px;
    color:rgba(255,255,255,0.4); text-transform:uppercase;
    background:rgba(0,0,0,0.65); padding:4px 8px; border-radius:4px;
    z-index:30;
}

/* IDLE PLACEHOLDER */
.idle-placeholder {
    display:flex; flex-direction:column; align-items:center; justify-content:center;
    height:380px; gap:1rem; z-index:10; position:relative;
}
.idle-hex {
    font-size:56px; opacity:0.08;
    animation: idle-float 4s ease-in-out infinite;
}
@keyframes idle-float {
    0%,100%{ transform:translateY(0) rotate(0deg); }
    50%    { transform:translateY(-10px) rotate(5deg); }
}
.idle-text {
    font-family:var(--mono); font-size:10px; letter-spacing:3px;
    color:rgba(255,255,255,0.12); text-transform:uppercase;
}
.orbit-ring {
    position:absolute; width:120px; height:120px;
    border:1px solid rgba(255,68,34,0.1); border-radius:50%;
    animation: orbit-spin 8s linear infinite;
}
.orbit-ring::before {
    content:''; position:absolute; top:-4px; left:50%;
    width:7px; height:7px; margin-left:-3.5px;
    background:var(--red); border-radius:50%;
    box-shadow:0 0 10px var(--red);
}
@keyframes orbit-spin {
    from { transform:rotate(0deg); }
    to   { transform:rotate(360deg); }
}

/* BUTTON */
[data-testid="stButton"] > button {
    width:100%; position:relative; overflow:hidden;
    background: linear-gradient(135deg, #ff4422 0%, #cc2200 100%);
    color:#fff; font-family:var(--mono); font-size:12px;
    font-weight:700; letter-spacing:4px; text-transform:uppercase;
    border:none; border-radius:10px; padding:17px 32px;
    cursor:pointer;
    box-shadow: 0 4px 24px rgba(255,68,34,0.35),
                inset 0 1px 0 rgba(255,255,255,0.12);
    transition: box-shadow .2s, transform .15s;
    z-index:10;
}
[data-testid="stButton"] > button::before {
    content:''; position:absolute; inset:0;
    background: linear-gradient(90deg,
        transparent 0%, rgba(255,255,255,0.22) 50%, transparent 100%);
    transform:translateX(-100%);
    transition: transform .55s ease;
}
[data-testid="stButton"] > button:hover::before { transform:translateX(100%); }
[data-testid="stButton"] > button:hover {
    box-shadow: 0 8px 36px rgba(255,68,34,0.55),
                inset 0 1px 0 rgba(255,255,255,0.12);
    transform: translateY(-2px);
}
[data-testid="stButton"] > button:active { transform:translateY(0); }

/* META CHIPS */
.meta-row { display:flex; gap:10px; margin-top:10px; z-index:10; position:relative; }
.meta-chip {
    flex:1; background:rgba(255,255,255,0.03);
    border:1px solid var(--border); border-radius:9px; padding:10px 12px;
}
.meta-chip-label {
    font-family:var(--mono); font-size:8px; letter-spacing:2px;
    color:rgba(255,255,255,0.22); text-transform:uppercase; margin-bottom:3px;
}
.meta-chip-value {
    font-family:var(--mono); font-size:12px; color:rgba(255,255,255,0.65); font-weight:700;
}

/* PROGRESS BAR */
[data-testid="stProgress"] > div {
    background:rgba(255,255,255,0.05) !important;
    height:3px !important; border-radius:100px !important;
}
[data-testid="stProgress"] > div > div {
    background:linear-gradient(90deg, var(--red), #ff8060) !important;
    border-radius:100px !important;
    box-shadow: 0 0 8px rgba(255,68,34,0.6) !important;
    transition: width .4s ease !important;
}

/* RESULT PANEL */
.result-panel {
    background:var(--panel);
    border:1px solid var(--border);
    border-radius:16px; padding:2rem;
    margin-top:1.4rem; position:relative; overflow:hidden;
    animation: panel-in 0.5s cubic-bezier(0.16,1,0.3,1) both;
    z-index:10;
}
@keyframes panel-in {
    from { opacity:0; transform:translateY(16px) scale(0.98); }
    to   { opacity:1; transform:translateY(0) scale(1); }
}
.result-panel::before {
    content:''; position:absolute; top:0; left:-100%; right:0; height:1px;
    background: linear-gradient(90deg, transparent, var(--red), transparent);
    animation: shimmer-line 2.5s ease-in-out infinite;
}
@keyframes shimmer-line {
    0%   { left:-100%; opacity:0.5; }
    100% { left:100%;  opacity:0.5; }
}

.result-eyebrow {
    font-family:var(--mono); font-size:9px; letter-spacing:4px;
    color:rgba(255,255,255,0.2); text-transform:uppercase; margin-bottom:1.4rem;
}

/* Glitch label */
.glitch-label {
    font-size:38px; font-weight:800; letter-spacing:-1.5px; line-height:1;
    position:relative; display:inline-block; margin-bottom:.3rem;
}
.glitch-label.defect { color:var(--red); }
.glitch-label.ok     { color:var(--green); }
.glitch-label.invalid { color:#ff9933; }
.glitch-label::before, .glitch-label::after {
    content: attr(data-text);
    position:absolute; top:0; left:0; right:0;
    overflow:hidden; clip:rect(0,0,0,0);
}
.glitch-label.defect::before {
    color:#ff0000; text-shadow:2px 0 #00ffff;
    animation: glitch-a 2.5s steps(1) infinite;
}
.glitch-label.defect::after {
    color:#ff0000; text-shadow:-2px 0 #ff00ff;
    animation: glitch-b 2.5s steps(1) infinite;
}
.glitch-label.ok::before {
    color:var(--green); text-shadow:2px 0 rgba(0,255,170,0.5);
    animation: glitch-a 4s steps(1) infinite;
}
.glitch-label.ok::after { display:none; }
.glitch-label.invalid::before {
    color:#ff9933; text-shadow:2px 0 rgba(255,153,51,0.5);
    animation: glitch-a 3s steps(1) infinite;
}
.glitch-label.invalid::after { display:none; }

@keyframes glitch-a {
    0%,90%,100%{ clip:rect(0,9999px,0,0); transform:translate(0); }
    91%         { clip:rect(14px,9999px,26px,0); transform:translate(-3px,0); }
    93%         { clip:rect(40px,9999px,52px,0); transform:translate(3px,0); }
    95%         { clip:rect(2px,9999px,8px,0);   transform:translate(-2px,0); }
}
@keyframes glitch-b {
    0%,88%,100%{ clip:rect(0,9999px,0,0); transform:translate(0); }
    89%         { clip:rect(30px,9999px,42px,0); transform:translate(3px,0); }
    91%         { clip:rect(8px,9999px,18px,0);  transform:translate(-3px,0); }
}

.result-sub {
    font-family:var(--mono); font-size:9px; letter-spacing:2px;
    color:rgba(255,255,255,0.25); text-transform:uppercase; margin-bottom:1.8rem;
}

/* Confidence bar */
.conf-track {
    background:rgba(255,255,255,0.05);
    border-radius:100px; height:6px;
    overflow:hidden; position:relative; margin:.4rem 0;
}
.conf-fill {
    height:100%; border-radius:100px;
    transition: width 1.2s cubic-bezier(0.16,1,0.3,1);
    position:relative;
}
.conf-fill.defect {
    background: linear-gradient(90deg, #cc2200, var(--red), #ff8060);
    box-shadow: 0 0 12px rgba(255,68,34,0.7);
}
.conf-fill.ok {
    background: linear-gradient(90deg, #00aa66, var(--green));
    box-shadow: 0 0 12px rgba(0,255,170,0.5);
}
.conf-fill.invalid {
    background: linear-gradient(90deg, #cc7700, #ff9933);
    box-shadow: 0 0 12px rgba(255,153,51,0.5);
}
.conf-fill::after {
    content:''; position:absolute; top:0; right:0; bottom:0; width:40px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.5), transparent);
    animation: conf-shimmer 2s ease-in-out infinite;
}
@keyframes conf-shimmer {
    0%   { transform:translateX(-40px); opacity:0; }
    50%  { opacity:1; }
    100% { transform:translateX(40px);  opacity:0; }
}

.conf-ticks {
    display:flex; justify-content:space-between;
    font-family:var(--mono); font-size:8px; color:rgba(255,255,255,0.18);
    margin-top:4px; letter-spacing:1px;
}

/* Stats row */
.stats-row {
    display:flex; gap:10px; margin-top:1.6rem; padding-top:1.4rem;
    border-top:1px solid rgba(255,255,255,0.05);
}
.stat-b {
    flex:1; background:rgba(255,255,255,0.03);
    border:1px solid var(--border); border-radius:10px; padding:11px 13px;
    transition: border-color .3s, background .3s;
}
.stat-b:hover {
    border-color:rgba(255,68,34,0.25);
    background:rgba(255,68,34,0.03);
}
.stat-b-lbl {
    font-family:var(--mono); font-size:8px; letter-spacing:2px;
    color:rgba(255,255,255,0.22); text-transform:uppercase; margin-bottom:4px;
}
.stat-b-val {
    font-family:var(--mono); font-size:14px; color:rgba(255,255,255,0.7); font-weight:700;
}

/* Scan status text */
.scan-status {
    font-family:var(--mono); font-size:10px; letter-spacing:3px;
    color:rgba(255,68,34,0.75); text-align:center;
    padding:.75rem; text-transform:uppercase;
}
.scan-status span { animation:blink-char .8s step-end infinite; }
@keyframes blink-char { 0%,100%{ opacity:1; } 50%{ opacity:0; } }

/* Status badges */
.uncertain-badge {
    display:inline-flex; align-items:center; gap:6px;
    background:rgba(255,180,0,0.07);
    border:1px solid rgba(255,180,0,0.2);
    border-radius:8px; padding:8px 14px;
    font-family:var(--mono); font-size:10px;
    color:rgba(255,180,0,0.75);
    letter-spacing:2px; text-transform:uppercase;
    margin-bottom:1rem; animation: panel-in .5s ease both;
}
.invalid-badge {
    display:inline-flex; align-items:center; gap:6px;
    background:rgba(255,100,0,0.07);
    border:1px solid rgba(255,100,0,0.25);
    border-radius:8px; padding:8px 14px;
    font-family:var(--mono); font-size:10px;
    color:rgba(255,120,50,0.9);
    letter-spacing:2px; text-transform:uppercase;
    margin-bottom:1rem; animation: panel-in .5s ease both;
}

/* HISTORY */
.hist-hdr {
    font-family:var(--mono); font-size:9px; letter-spacing:4px;
    color:rgba(255,255,255,0.2); text-transform:uppercase;
    margin-bottom:.9rem; padding-bottom:.75rem;
    border-bottom:1px solid var(--border);
    z-index:10; position:relative;
}
.hist-item {
    display:flex; align-items:center; gap:12px;
    padding:9px 0; border-bottom:1px solid rgba(255,255,255,0.04);
    animation: slide-in .35s ease both;
    z-index:10; position:relative;
}
@keyframes slide-in {
    from{ opacity:0; transform:translateX(-10px); }
    to  { opacity:1; transform:translateX(0); }
}
.hist-dot { width:8px; height:8px; border-radius:50%; flex-shrink:0; }
.hist-dot.defect  { background:var(--red);   box-shadow:0 0 7px var(--red); }
.hist-dot.ok      { background:var(--green); box-shadow:0 0 7px var(--green); }
.hist-dot.invalid { background:#ff9933;      box-shadow:0 0 7px #ff9933; }
.hist-text { font-family:var(--mono); font-size:10px; color:rgba(255,255,255,0.45); flex:1; }
.hist-conf { font-family:var(--mono); font-size:10px; color:rgba(255,255,255,0.22); }

/* ERROR BOX */
.err-box {
    background:rgba(255,40,40,0.05);
    border:1px solid rgba(255,40,40,0.15);
    border-radius:10px; padding:1.5rem; text-align:center;
    animation: panel-in .4s ease both; z-index:10; position:relative;
}
.err-title { font-family:var(--mono); font-size:10px; letter-spacing:2px; color:rgba(255,80,80,.75); text-transform:uppercase; }
.err-sub   { font-family:var(--mono); font-size:9px; color:rgba(255,255,255,.2); margin-top:5px; }

/* FOOTER */
.footer {
    font-family:var(--mono); font-size:9px; letter-spacing:2px;
    color:rgba(255,255,255,.12); text-align:center;
    padding:2rem 0 1rem; text-transform:uppercase;
    z-index:10; position:relative;
}
.footer em { color:var(--red); font-style:normal; }

hr { border:none !important; border-top:1px solid var(--border) !important; margin:2rem 0 !important; }
[data-testid="stColumns"] { gap:2.2rem !important; }
::-webkit-scrollbar { width:4px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:rgba(255,255,255,.09); border-radius:10px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────── Session state ───────────────────────
if "history"    not in st.session_state: st.session_state.history    = []
if "scan_count" not in st.session_state: st.session_state.scan_count = 0

def fmt_class(name: str) -> str:
    """Convert 'rolled-in_scale' → 'Rolled-In Scale', 'pitted_surface' → 'Pitted Surface'"""
    return " ".join(
        word.capitalize()
        for word in name.replace("-", " - ").replace("_", " ").split()
    ).replace(" - ", "-")

# ─────────────────────────── Header ───────────────────────────
now = datetime.now().strftime("%Y.%m.%d  %H:%M")
st.markdown(f"""
<div class="hdr">
    <div>
        <div class="brand-eye">⚡ Surface Defect Detection System</div>
        <div class="brand-name">Defect<em>Scan</em></div>
        <div class="brand-sub">EfficientNet-B4 &nbsp;·&nbsp; FastAPI &nbsp;·&nbsp; TensorFlow 2.x &nbsp;·&nbsp; v2.4.0</div>
    </div>
    <div class="live-badge">
        <div class="live-dot"></div>
        LIVE &nbsp;·&nbsp; {now}
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────── Two-column layout ───────────────────────
col1, col2 = st.columns([1.15, 0.85])

with col1:
    uploaded_file = st.file_uploader(
        "DROP IMAGE TO ANALYZE  ·  JPG / PNG / JPEG",
        type=["jpg", "jpeg", "png"],
        label_visibility="visible"
    )

    if uploaded_file:
        # Framed image with sweep + reticle
        st.markdown("""
        <div class="img-frame">
            <div class="c c-tl"></div><div class="c c-tr"></div>
            <div class="c c-bl"></div><div class="c c-br"></div>
            <div class="sweep"></div>
            <div class="reticle"></div>
            <div class="img-label">INPUT · SURFACE SCAN</div>
        """, unsafe_allow_html=True)
        st.image(uploaded_file, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Meta chips
        kb    = len(uploaded_file.getvalue()) / 1024
        fmt   = uploaded_file.type.split("/")[-1].upper()
        name  = uploaded_file.name
        short = (name[:20] + "…") if len(name) > 20 else name
        st.markdown(f"""
        <div class="meta-row">
            <div class="meta-chip">
                <div class="meta-chip-label">File</div>
                <div class="meta-chip-value">{short}</div>
            </div>
            <div class="meta-chip">
                <div class="meta-chip-label">Size</div>
                <div class="meta-chip-value">{kb:.1f} KB</div>
            </div>
            <div class="meta-chip">
                <div class="meta-chip-label">Format</div>
                <div class="meta-chip-value">{fmt}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

with col2:
    if not uploaded_file:
        st.markdown("""
        <div class="idle-placeholder">
            <div class="orbit-ring"></div>
            <div class="idle-hex">⬡</div>
            <div class="idle-text">Awaiting surface image</div>
        </div>
        """, unsafe_allow_html=True)

    if uploaded_file:
        run_btn = st.button("⚡  INITIATE SCAN")

        if run_btn:
            st.session_state.scan_count += 1
            scan_start = time.time()

            pb     = st.progress(0)
            status = st.empty()

            steps = [
                (8,  "[ SYS ] Decoding image buffer…"),
                (22, "[ NET ] Loading EfficientNet-B4 weights…"),
                (44, "[ GPU ] Running forward pass…"),
                (68, "[ OUT ] Computing softmax distribution…"),
                (85, "[ POST] Applying confidence threshold…"),
                (97, "[ FIN ] Serializing prediction output…"),
            ]
            for pct, msg in steps:
                pb.progress(pct)
                status.markdown(
                    f'<div class="scan-status">▶ {msg} <span>█</span></div>',
                    unsafe_allow_html=True
                )
                time.sleep(0.22)

            try:
                files    = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                response = requests.post(API_URL, files=files, timeout=15)
                scan_time = time.time() - scan_start
                pb.progress(100)
                status.empty()

                if response.status_code == 200:
                    result      = response.json()
                    confidence  = result.get("confidence", 0)
                    pred_class     = result.get("predicted_class", "UNKNOWN")
                    pred_class_fmt = fmt_class(pred_class)
                    status_code = result.get("status", "uncertain")

                    # ── Determine visual class based on status ──
                    if status_code == "success":
                        is_defect = "defect" in pred_class.lower()
                        cls       = "defect" if is_defect else "ok"
                        icon      = "⛔" if is_defect else "✅"
                        hist_type = cls
                    elif status_code == "invalid_input":
                        cls       = "invalid"
                        icon      = "⚠"
                        hist_type = "invalid"
                    else:
                        cls       = "defect"   # neutral colour for uncertain
                        icon      = "?"
                        hist_type = "defect"

                    conf_pct = int(confidence * 100)

                    st.session_state.history.insert(0, {
                        "class": pred_class_fmt, "confidence": confidence,
                        "type": hist_type, "time": datetime.now().strftime("%H:%M:%S")
                    })
                    if len(st.session_state.history) > 6:
                        st.session_state.history.pop()

                    # ── Result panel ──
                    st.markdown('<div class="result-panel">', unsafe_allow_html=True)
                    st.markdown('<div class="result-eyebrow">// scan output</div>', unsafe_allow_html=True)

                    # ── Three-way branching for status ──
                    if status_code == "success":
                        st.markdown(f"""
                        <div class="glitch-label {cls}" data-text="{icon} {pred_class_fmt}">{icon} {pred_class_fmt}</div>
                        <div class="result-sub">Classification · EfficientNet-B4 · threshold 0.85</div>
                        """, unsafe_allow_html=True)

                    elif status_code == "invalid_input":
                        error_msg = result.get("message", "Input rejected by model")
                        st.markdown(f"""
                        <div class="invalid-badge">⛔ INVALID INPUT — {error_msg}</div>
                        <div class="glitch-label {cls}" data-text="{icon} {pred_class_fmt}">{icon} {pred_class_fmt}</div>
                        <div class="result-sub">Classification · EfficientNet-B4 · threshold 0.85</div>
                        """, unsafe_allow_html=True)

                    else:
                        # uncertain
                        st.markdown("""
                        <div class="uncertain-badge">⚠ LOW CONFIDENCE — MODEL UNCERTAIN</div>
                        """, unsafe_allow_html=True)

                    # ── Confidence bar — unique animation name per scan ──
                    anim_name  = f"fill_{st.session_state.scan_count}"
                    conf_color = (
                        "#ff4422" if status_code == "success" and cls == "defect"
                        else "#ff9933" if status_code == "invalid_input"
                        else "#00ffaa"
                    )
                    st.markdown(f"""
                    <style>
                    @keyframes {anim_name} {{
                        0%   {{ width: 0%; }}
                        100% {{ width: {conf_pct}%; }}
                    }}
                    .conf-fill-live {{
                        height: 100%;
                        border-radius: 100px;
                        animation: {anim_name} 1.2s cubic-bezier(0.16,1,0.3,1) forwards;
                        position: relative;
                    }}
                    .conf-fill-live.defect {{
                        background: linear-gradient(90deg, #cc2200, #ff4422, #ff8060);
                        box-shadow: 0 0 12px rgba(255,68,34,0.7);
                    }}
                    .conf-fill-live.ok {{
                        background: linear-gradient(90deg, #00aa66, #00ffaa);
                        box-shadow: 0 0 12px rgba(0,255,170,0.5);
                    }}
                    .conf-fill-live.invalid {{
                        background: linear-gradient(90deg, #cc7700, #ff9933);
                        box-shadow: 0 0 12px rgba(255,153,51,0.5);
                    }}
                    </style>
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                        <span style="font-family:'Space Mono',monospace;font-size:9px;letter-spacing:2px;
                              color:rgba(255,255,255,.22);text-transform:uppercase;">Confidence</span>
                        <span style="font-family:'Space Mono',monospace;font-size:28px;font-weight:700;
                              color:{conf_color};">{conf_pct}<span style="font-size:14px;color:rgba(255,255,255,0.35);">%</span></span>
                    </div>
                    <div class="conf-track">
                        <div class="conf-fill-live {cls}"></div>
                    </div>
                    <div class="conf-ticks">
                        <span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── Stats row ──
                    st.markdown(f"""
                    <div class="stats-row">
                        <div class="stat-b">
                            <div class="stat-b-lbl">Raw score</div>
                            <div class="stat-b-val">{confidence:.4f}</div>
                        </div>
                        <div class="stat-b">
                            <div class="stat-b-lbl">Scan time</div>
                            <div class="stat-b-val">{scan_time:.2f}s</div>
                        </div>
                        <div class="stat-b">
                            <div class="stat-b-lbl">Session</div>
                            <div class="stat-b-val">#{st.session_state.scan_count}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown('</div>', unsafe_allow_html=True)

                else:
                    pb.empty()
                    st.markdown(f"""<div class="err-box">
                        <div class="err-title">API ERROR {response.status_code}</div>
                        <div class="err-sub">Backend returned an unexpected status code.</div>
                    </div>""", unsafe_allow_html=True)

            except requests.exceptions.ConnectionError:
                pb.empty(); status.empty()
                st.markdown("""<div class="err-box">
                    <div style="font-size:22px;margin-bottom:8px;">⚡</div>
                    <div class="err-title">Cannot reach API at 127.0.0.1:8000</div>
                    <div class="err-sub">Make sure your FastAPI server is running.</div>
                </div>""", unsafe_allow_html=True)

            except Exception as e:
                pb.empty(); status.empty()
                st.markdown(f"""<div class="err-box">
                    <div style="font-size:22px;margin-bottom:8px;">⚠</div>
                    <div class="err-title">Unexpected Error</div>
                    <div class="err-sub">{e}</div>
                </div>""", unsafe_allow_html=True)

# ─────────────────────── Scan History ───────────────────────
if st.session_state.history:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="hist-hdr">// scan history</div>', unsafe_allow_html=True)
    for i, item in enumerate(st.session_state.history):
        st.markdown(f"""
        <div class="hist-item" style="animation-delay:{i*0.06}s;">
            <div class="hist-dot {item['type']}"></div>
            <div class="hist-text">{item['class']}</div>
            <div class="hist-conf">{item['confidence']:.3f} &nbsp;·&nbsp; {item['time']}</div>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────── Footer ───────────────────────────
st.markdown("""
<div class="footer">
    DefectScan AI &nbsp;·&nbsp; Built with <em>TensorFlow</em> &nbsp;·&nbsp; FastAPI &nbsp;·&nbsp; Streamlit
</div>
""", unsafe_allow_html=True)