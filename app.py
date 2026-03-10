import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from gesture_engine import GestureEngine
import time

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="GestureVision",
    page_icon="🖐",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0f;
    color: #e8e8f0;
    font-family: 'DM Mono', monospace;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 20% 10%, #1a1040 0%, #0a0a0f 60%),
                radial-gradient(ellipse at 80% 90%, #0d1f2d 0%, transparent 60%);
    min-height: 100vh;
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* Header */
.gesture-header {
    display: flex;
    align-items: flex-end;
    gap: 16px;
    padding: 40px 0 8px 0;
    border-bottom: 1px solid #1e1e2e;
    margin-bottom: 32px;
}
.gesture-title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -2px;
    background: linear-gradient(135deg, #a78bfa, #38bdf8, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
}
.gesture-subtitle {
    font-size: 0.7rem;
    color: #4a4a6a;
    letter-spacing: 4px;
    text-transform: uppercase;
    padding-bottom: 6px;
}

/* Stat cards */
.stat-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-bottom: 24px;
}
.stat-card {
    background: #0f0f1a;
    border: 1px solid #1e1e32;
    border-radius: 8px;
    padding: 16px 20px;
    position: relative;
    overflow: hidden;
}
.stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #a78bfa, #38bdf8);
}
.stat-label {
    font-size: 0.6rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #4a4a6a;
    margin-bottom: 6px;
}
.stat-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #e8e8f0;
}

/* Gesture display box */
.gesture-display {
    background: #0f0f1a;
    border: 1px solid #1e1e32;
    border-radius: 8px;
    padding: 24px;
    text-align: center;
    min-height: 120px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    margin-bottom: 16px;
    position: relative;
    overflow: hidden;
}
.gesture-display::after {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at center, #a78bfa08, transparent 70%);
    pointer-events: none;
}
.gesture-emoji {
    font-size: 3.5rem;
    line-height: 1;
    margin-bottom: 8px;
}
.gesture-name {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #a78bfa;
    letter-spacing: -0.5px;
}
.gesture-meaning {
    font-size: 0.72rem;
    color: #4a4a6a;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 4px;
}

/* Control button */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #a78bfa22, #38bdf822) !important;
    border: 1px solid #a78bfa55 !important;
    color: #a78bfa !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    padding: 12px 24px !important;
    border-radius: 6px !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #a78bfa44, #38bdf844) !important;
    border-color: #a78bfa99 !important;
    color: #c4b5fd !important;
}

/* Gesture log */
.log-container {
    background: #0f0f1a;
    border: 1px solid #1e1e32;
    border-radius: 8px;
    padding: 16px;
    max-height: 200px;
    overflow-y: auto;
}
.log-entry {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 6px 0;
    border-bottom: 1px solid #1a1a2e;
    font-size: 0.72rem;
}
.log-entry:last-child { border-bottom: none; }
.log-time { color: #4a4a6a; min-width: 60px; }
.log-gesture { color: #a78bfa; font-weight: 500; }
.log-action { color: #38bdf8; }

/* Section label */
.section-label {
    font-size: 0.6rem;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: #4a4a6a;
    margin-bottom: 10px;
}

/* Confidence bar */
.conf-bar-bg {
    background: #1a1a2e;
    border-radius: 2px;
    height: 4px;
    margin-top: 8px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 2px;
    background: linear-gradient(90deg, #a78bfa, #38bdf8);
    transition: width 0.3s ease;
}

/* Camera frame */
[data-testid="stImage"] img {
    border-radius: 8px;
    border: 1px solid #1e1e32;
}
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────
if "running" not in st.session_state:
    st.session_state.running = False
if "gesture_log" not in st.session_state:
    st.session_state.gesture_log = []
if "total_detected" not in st.session_state:
    st.session_state.total_detected = 0
if "current_gesture" not in st.session_state:
    st.session_state.current_gesture = None

engine = GestureEngine()

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="gesture-header">
    <div>
        <div class="gesture-title">GestureVision</div>
    </div>
    <div class="gesture-subtitle">Real-time Hand Gesture Recognition &nbsp;/&nbsp; MediaPipe + OpenCV</div>
</div>
""", unsafe_allow_html=True)

# ── Layout ────────────────────────────────────────────────────
col_cam, col_info = st.columns([3, 2], gap="large")

with col_cam:
    st.markdown('<div class="section-label">Live Camera Feed</div>', unsafe_allow_html=True)
    frame_placeholder = st.empty()
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        start_btn = st.button("START DETECTION", key="start")
    with btn_col2:
        stop_btn = st.button("STOP DETECTION", key="stop")

with col_info:
    # Stats
    stat_placeholder = st.empty()

    # Current gesture display
    st.markdown('<div class="section-label">Detected Gesture</div>', unsafe_allow_html=True)
    gesture_placeholder = st.empty()

    # Gesture log
    st.markdown('<div class="section-label">Detection Log</div>', unsafe_allow_html=True)
    log_placeholder = st.empty()

# ── Render helpers ────────────────────────────────────────────
def render_stats(total, fps, confidence):
    stat_placeholder.markdown(f"""
    <div class="stat-row">
        <div class="stat-card">
            <div class="stat-label">Detected</div>
            <div class="stat-value">{total:04d}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">FPS</div>
            <div class="stat-value">{fps:.0f}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Confidence</div>
            <div class="stat-value">{confidence:.0f}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_gesture(gesture_info):
    if gesture_info:
        gesture_placeholder.markdown(f"""
        <div class="gesture-display">
            <div class="gesture-emoji">{gesture_info['emoji']}</div>
            <div class="gesture-name">{gesture_info['name']}</div>
            <div class="gesture-meaning">{gesture_info['meaning']}</div>
            <div class="conf-bar-bg">
                <div class="conf-bar-fill" style="width:{gesture_info['confidence']}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        gesture_placeholder.markdown("""
        <div class="gesture-display">
            <div class="gesture-meaning">Waiting for gesture...</div>
        </div>
        """, unsafe_allow_html=True)

def render_log(log):
    if not log:
        log_placeholder.markdown("""
        <div class="log-container">
            <div class="log-entry"><span class="log-time">--:--</span><span class="log-gesture">No detections yet</span></div>
        </div>
        """, unsafe_allow_html=True)
        return
    entries = ""
    for entry in reversed(log[-8:]):
        entries += f"""
        <div class="log-entry">
            <span class="log-time">{entry['time']}</span>
            <span class="log-gesture">{entry['emoji']} {entry['name']}</span>
            <span class="log-action">→ {entry['meaning']}</span>
        </div>"""
    log_placeholder.markdown(f'<div class="log-container">{entries}</div>', unsafe_allow_html=True)

# ── Initial render ────────────────────────────────────────────
render_stats(st.session_state.total_detected, 0, 0)
render_gesture(None)
render_log(st.session_state.gesture_log)

frame_placeholder.markdown("""
<div style="background:#0f0f1a;border:1px solid #1e1e32;border-radius:8px;
height:360px;display:flex;align-items:center;justify-content:center;
flex-direction:column;gap:12px;">
    <div style="font-size:2.5rem;opacity:0.3">🖐</div>
    <div style="font-size:0.65rem;letter-spacing:4px;text-transform:uppercase;color:#4a4a6a">
        Press Start Detection
    </div>
</div>
""", unsafe_allow_html=True)

# ── Button logic ──────────────────────────────────────────────
if start_btn:
    st.session_state.running = True

if stop_btn:
    st.session_state.running = False

# ── Detection Loop ────────────────────────────────────────────
if st.session_state.running:
    cap = None
    # Try different camera indices and wait for the camera to warm up
    for cam_idx in [0, 1]:
        temp_cap = cv2.VideoCapture(cam_idx)
        if temp_cap.isOpened():
            # Try a few reads to let the camera warm up
            for _ in range(5):
                ret, _ = temp_cap.read()
                if ret:
                    cap = temp_cap
                    break
                time.sleep(0.1)
        if cap is not None:
            break
        temp_cap.release()

    if cap is None or not cap.isOpened():
        st.error("Camera not found. Please ensure a webcam is connected and the app has camera permissions.")
        st.session_state.running = False
        st.stop()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    prev_time = time.time()
    prev_gesture = None

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to grab frame. Retrying...")
            time.sleep(0.1)
            continue

        frame = cv2.flip(frame, 1)
        result_frame, gesture_info = engine.process(frame)

        # FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time + 1e-9)
        prev_time = curr_time

        # Log new gestures
        if gesture_info and gesture_info["name"] != prev_gesture:
            st.session_state.total_detected += 1
            st.session_state.gesture_log.append({
                "time": time.strftime("%H:%M"),
                "emoji": gesture_info["emoji"],
                "name": gesture_info["name"],
                "meaning": gesture_info["meaning"]
            })
            prev_gesture = gesture_info["name"]

        # Render
        rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(rgb, use_container_width=True)
        render_stats(st.session_state.total_detected, fps, gesture_info["confidence"] if gesture_info else 0)
        render_gesture(gesture_info)
        render_log(st.session_state.gesture_log)

    cap.release()
