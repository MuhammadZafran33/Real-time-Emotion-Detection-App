import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
import pandas as pd
import matplotlib.pyplot as plt
import time
from collections import deque
from datetime import datetime

st.set_page_config(
    page_title="EmotionAI — Real-time Detection",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@600;700;800&display=swap');

* { box-sizing: border-box; }
html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    background: #07070f !important;
    color: #e2e2ef !important;
}
.block-container { padding: 1.5rem 2.5rem !important; max-width: 1400px !important; }
#MainMenu, footer, header { visibility: hidden !important; }

/* BG orbs */
.orb { position: fixed; border-radius: 50%; filter: blur(120px); pointer-events: none; z-index: 0; }
.o1 { width:500px;height:500px;background:rgba(99,102,241,0.1);top:-150px;left:-150px;animation:drift 12s ease-in-out infinite; }
.o2 { width:400px;height:400px;background:rgba(236,72,153,0.08);bottom:-100px;right:-100px;animation:drift 10s ease-in-out infinite reverse; }
.o3 { width:350px;height:350px;background:rgba(16,185,129,0.07);top:40%;left:35%;animation:drift 14s ease-in-out infinite 3s; }
@keyframes drift { 0%,100%{transform:translate(0,0)} 33%{transform:translate(30px,-30px)} 66%{transform:translate(-20px,20px)} }

/* Hero */
.hero { text-align:center; padding:2rem 1rem 1.5rem; position:relative; z-index:1; }
.hero-badge {
    display:inline-flex; align-items:center; gap:8px;
    background:rgba(99,102,241,0.12); border:1px solid rgba(99,102,241,0.35);
    color:#a5b4fc; font-size:11px; font-weight:700; letter-spacing:3px; text-transform:uppercase;
    padding:7px 22px; border-radius:100px; margin-bottom:1.2rem;
}
.hero-badge .dot { width:7px;height:7px;background:#818cf8;border-radius:50%;animation:blink 1.5s infinite; }
@keyframes blink { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.2;transform:scale(0.6)} }
.hero-title {
    font-family:'Space Grotesk',sans-serif;
    font-size:3.2rem; font-weight:800; line-height:1.1;
    letter-spacing:-1.5px; color:#f1f1f8; margin-bottom:0.8rem;
}
.grad { background:linear-gradient(135deg,#818cf8 0%,#ec4899 50%,#f59e0b 100%);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }
.hero-sub { font-size:1rem; color:#6b6b8a; max-width:500px; margin:0 auto 1.5rem; line-height:1.7; }

/* Emotion cards */
.emo-card {
    background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07);
    border-radius:20px; padding:20px; text-align:center;
    transition:all 0.3s ease; position:relative; overflow:hidden;
}
.emo-card.active {
    background:rgba(99,102,241,0.1); border-color:rgba(99,102,241,0.4);
    box-shadow:0 8px 32px rgba(99,102,241,0.2);
    transform:translateY(-4px);
}
.emo-icon { font-size:2.8rem; display:block; margin-bottom:8px; }
.emo-name { font-size:12px; font-weight:700; letter-spacing:2px; text-transform:uppercase; color:#6b7280; }
.emo-name.active { color:#a5b4fc; }
.emo-pct { font-family:'Space Grotesk',sans-serif; font-size:1.6rem; font-weight:800; color:#e2e2ef; margin-top:4px; }

/* Status bar */
.status-bar {
    display:flex; align-items:center; gap:12px;
    background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07);
    border-radius:14px; padding:14px 20px; margin-bottom:16px;
}
.status-dot-live { width:10px;height:10px;background:#ef4444;border-radius:50%;animation:blink 1s infinite;flex-shrink:0; }
.status-dot-off { width:10px;height:10px;background:#374151;border-radius:50%;flex-shrink:0; }
.status-text { font-size:13px; font-weight:600; color:#9ca3af; }

/* Result hero */
.result-hero {
    border-radius:20px; padding:2rem; text-align:center;
    animation:popIn 0.4s cubic-bezier(0.34,1.56,0.64,1);
    margin-bottom:16px;
}
@keyframes popIn { from{opacity:0;transform:scale(0.8)} to{opacity:1;transform:scale(1)} }

/* Metric glass */
.mg { background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07); border-radius:16px; padding:18px; text-align:center; }
.mg-label { font-size:10px;font-weight:800;letter-spacing:2.5px;text-transform:uppercase;color:#818cf8;margin-bottom:8px; }
.mg-val { font-family:'Space Grotesk',sans-serif;font-size:1.8rem;font-weight:800;color:#e2e2ef;line-height:1; }

/* Buttons */
.stButton > button {
    border-radius:12px !important; font-weight:700 !important;
    font-size:14px !important; transition:all 0.25s !important;
}
.stButton > button[kind="primary"] {
    background:linear-gradient(135deg,#6366f1,#8b5cf6) !important;
    color:white !important; border:none !important;
    padding:14px 28px !important;
    box-shadow:0 6px 24px rgba(99,102,241,0.4) !important;
}
.stButton > button[kind="primary"]:hover { transform:translateY(-2px) !important; }
.stButton > button:not([kind="primary"]) {
    background:rgba(239,68,68,0.12) !important;
    border:1px solid rgba(239,68,68,0.35) !important;
    color:#f87171 !important; padding:14px 28px !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background:rgba(255,255,255,0.03) !important; border-radius:12px !important;
    padding:4px !important; border:1px solid rgba(255,255,255,0.06) !important;
}
.stTabs [data-baseweb="tab"] { border-radius:8px !important; color:#555572 !important; font-weight:600 !important; padding:10px 24px !important; }
.stTabs [aria-selected="true"] { background:rgba(99,102,241,0.18) !important; color:#a5b4fc !important; }
.stTabs [data-baseweb="tab-border"] { display:none !important; }

/* Metrics */
[data-testid="stMetric"] { background:rgba(255,255,255,0.03) !important; border:1px solid rgba(255,255,255,0.07) !important; border-radius:14px !important; padding:16px !important; }
[data-testid="stMetricLabel"] { color:#818cf8 !important; font-size:11px !important; font-weight:700 !important; letter-spacing:1.5px !important; text-transform:uppercase !important; }
[data-testid="stMetricValue"] { color:#e2e2ef !important; font-family:'Space Grotesk',sans-serif !important; font-size:1.8rem !important; }

/* Footer */
.footer { text-align:center; padding:2rem 0 1rem; border-top:1px solid rgba(255,255,255,0.05); margin-top:2rem; color:#2a2a40; font-size:13px; }
.footer a { color:#6366f1; text-decoration:none; font-weight:700; }
hr { border-color:rgba(255,255,255,0.05) !important; }
</style>

<div class="orb o1"></div>
<div class="orb o2"></div>
<div class="orb o3"></div>
""", unsafe_allow_html=True)

# ── Emotion config ────────────────────────────────────────
EMOTIONS = {
    "happy":     {"emoji": "😊", "color": "#f59e0b", "label": "Happy"},
    "sad":       {"emoji": "😢", "color": "#3b82f6", "label": "Sad"},
    "angry":     {"emoji": "😠", "color": "#ef4444", "label": "Angry"},
    "surprise":  {"emoji": "😲", "color": "#8b5cf6", "label": "Surprise"},
    "fear":      {"emoji": "😨", "color": "#6366f1", "label": "Fear"},
    "disgust":   {"emoji": "🤢", "color": "#10b981", "label": "Disgust"},
    "neutral":   {"emoji": "😐", "color": "#6b7280", "label": "Neutral"},
}

# ── Session state ─────────────────────────────────────────
if "running"       not in st.session_state: st.session_state.running = False
if "history"       not in st.session_state: st.session_state.history = deque(maxlen=50)
if "frame_count"   not in st.session_state: st.session_state.frame_count = 0
if "dominant"      not in st.session_state: st.session_state.dominant = None
if "scores"        not in st.session_state: st.session_state.scores = {}
if "screenshots"   not in st.session_state: st.session_state.screenshots = []
if "total_frames"  not in st.session_state: st.session_state.total_frames = 0
if "session_start" not in st.session_state: st.session_state.session_start = None

# ── Hero ──────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge"><span class="dot"></span>Real-time AI · Computer Vision</div>
    <h1 class="hero-title">Detect Human Emotions<br><span class="grad">Live from Webcam</span></h1>
    <p class="hero-sub">AI-powered real-time facial emotion recognition — 7 emotions detected instantly using DeepFace.</p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "  🎥  Live Detection  ",
    "  📊  Session Analytics  ",
    "  📸  Screenshots  "
])

# ══════════════════════════════════════════════
# TAB 1 — Live Detection
# ══════════════════════════════════════════════
with tab1:
    st.markdown("<br>", unsafe_allow_html=True)

    # Control buttons
    bc1, bc2, bc3 = st.columns([1, 1, 2])
    with bc1:
        start_btn = st.button("▶  Start Detection", type="primary", use_container_width=True)
    with bc2:
        stop_btn = st.button("⏹  Stop", use_container_width=True)

    if start_btn:
        st.session_state.running = True
        st.session_state.session_start = datetime.now()
        st.session_state.history.clear()
        st.session_state.total_frames = 0

    if stop_btn:
        st.session_state.running = False

    st.markdown("<br>", unsafe_allow_html=True)

    # Status bar
    if st.session_state.running:
        st.markdown("""<div class="status-bar">
            <div class="status-dot-live"></div>
            <span class="status-text">🔴 LIVE — Detecting emotions in real-time</span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class="status-bar">
            <div class="status-dot-off"></div>
            <span class="status-text">⚫ STOPPED — Click Start Detection to begin</span>
        </div>""", unsafe_allow_html=True)

    # Main layout
    left, right = st.columns([1.4, 1], gap="large")

    with left:
        feed_placeholder = st.empty()

    with right:
        result_placeholder = st.empty()
        bars_placeholder   = st.empty()
        metrics_placeholder = st.empty()

    # Emotion grid
    st.markdown("<br>**🎭 Emotion Monitor**", unsafe_allow_html=False)
    grid_placeholder = st.empty()

    # ── Main detection loop ──────────────────────────────
    if st.session_state.running:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("❌ Cannot open webcam. Make sure your camera is connected and not used by another app.")
            st.session_state.running = False
        else:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            analyze_every = 3
            frame_idx = 0
            last_result = None

            while st.session_state.running:
                ret, frame = cap.read()
                if not ret:
                    st.warning("⚠️ Cannot read from webcam.")
                    break

                frame_idx += 1
                st.session_state.total_frames += 1
                display = frame.copy()

                # Detect faces
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80,80))

                # Draw face boxes
                for (x, y, w, h) in faces:
                    cv2.rectangle(display, (x,y), (x+w, y+h), (99,102,241), 2)
                    cv2.rectangle(display, (x, y-32), (x+w, y), (99,102,241), -1)
                    emotion_label = st.session_state.dominant or "Detecting..."
                    cv2.putText(display, emotion_label.upper(), (x+8, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

                # Analyze every N frames
                if frame_idx % analyze_every == 0 and len(faces) > 0:
                    try:
                        result = DeepFace.analyze(frame, actions=['emotion'],
                                                   enforce_detection=False, silent=True)
                        if isinstance(result, list): result = result[0]
                        emotions = result.get('emotion', {})
                        dominant = result.get('dominant_emotion', 'neutral')

                        st.session_state.scores = emotions
                        st.session_state.dominant = dominant
                        st.session_state.history.append({
                            'time': datetime.now().strftime('%H:%M:%S'),
                            'emotion': dominant,
                            **{k: round(v,1) for k,v in emotions.items()}
                        })
                        last_result = emotions
                    except Exception:
                        pass

                # Timestamp overlay
                ts = datetime.now().strftime('%H:%M:%S')
                cv2.putText(display, f"EmotionAI  {ts}", (10, display.shape[0]-12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (99,102,241), 1)
                cv2.putText(display, f"Faces: {len(faces)}", (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

                # Show frame
                rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                feed_placeholder.image(rgb, channels="RGB", use_container_width=True)

                # Update right panel
                dom = st.session_state.dominant
                scores = st.session_state.scores

                if dom and scores:
                    emo = EMOTIONS.get(dom, EMOTIONS["neutral"])
                    col_hex = emo['color']

                    with result_placeholder.container():
                        st.markdown(f"""
                        <div class="result-hero" style="background:linear-gradient(145deg,#111,#1a1a2e);
                             border:1px solid {col_hex}55; box-shadow:0 16px 40px {col_hex}22;">
                            <span style="font-size:72px;display:block;margin-bottom:10px">{emo['emoji']}</span>
                            <div style="font-family:'Space Grotesk',sans-serif;font-size:2rem;
                                 font-weight:800;color:{col_hex}">{emo['label'].upper()}</div>
                            <div style="font-size:13px;color:#6b7280;margin-top:6px">Dominant Emotion</div>
                        </div>""", unsafe_allow_html=True)

                    # Top 3 bars
                    with bars_placeholder.container():
                        sorted_emo = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
                        fig, ax = plt.subplots(figsize=(4, 2.2))
                        fig.patch.set_facecolor('#0d0d18')
                        ax.set_facecolor('#0d0d18')
                        labels = [EMOTIONS.get(e,{}).get('label',e) for e,_ in sorted_emo]
                        vals   = [v for _,v in sorted_emo]
                        colors = [EMOTIONS.get(e,{}).get('color','#6366f1') for e,_ in sorted_emo]
                        bars = ax.barh(labels, vals, color=colors, height=0.5, edgecolor='none')
                        for bar, val in zip(bars, vals):
                            ax.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
                                    f'{val:.1f}%', va='center', color='white', fontsize=10, fontweight='700')
                        ax.set_xlim(0, 105)
                        ax.tick_params(colors='#9ca3af', labelsize=11)
                        ax.set_xlabel('Confidence %', color='#4b5563', fontsize=9)
                        for sp in ax.spines.values(): sp.set_visible(False)
                        ax.yaxis.grid(False); ax.xaxis.grid(True, color='#1f2937', linewidth=0.6)
                        fig.tight_layout()
                        st.pyplot(fig)
                        plt.close()

                # Emotion grid
                with grid_placeholder.container():
                    cols = st.columns(7)
                    for col, (emo_key, emo_data) in zip(cols, EMOTIONS.items()):
                        pct = scores.get(emo_key, 0)
                        is_active = dom == emo_key
                        active_cls = "active" if is_active else ""
                        col.markdown(f"""
                        <div class="emo-card {active_cls}">
                            <span class="emo-icon">{emo_data['emoji']}</span>
                            <div class="emo-name {active_cls}">{emo_data['label']}</div>
                            <div class="emo-pct" style="color:{'#e2e2ef' if not is_active else emo_data['color']}">{pct:.0f}%</div>
                        </div>""", unsafe_allow_html=True)

                time.sleep(0.04)

            cap.release()

    else:
        # Placeholder when not running
        feed_placeholder.markdown("""
        <div style="background:rgba(255,255,255,0.02);border:2px dashed rgba(99,102,241,0.2);
             border-radius:16px;padding:5rem;text-align:center;color:#2a2a40;">
            <div style="font-size:64px;margin-bottom:16px;filter:grayscale(1) opacity(0.3)">🎥</div>
            <p style="font-size:16px;font-weight:600;color:#3a3a58">Camera feed will appear here</p>
            <p style="font-size:13px;margin-top:8px;color:#252540">Click Start Detection above</p>
        </div>""", unsafe_allow_html=True)

        with grid_placeholder.container():
            cols = st.columns(7)
            for col, (_, emo_data) in zip(cols, EMOTIONS.items()):
                col.markdown(f"""
                <div class="emo-card">
                    <span class="emo-icon">{emo_data['emoji']}</span>
                    <div class="emo-name">{emo_data['label']}</div>
                    <div class="emo-pct" style="color:#2a2a40">—</div>
                </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 2 — Session Analytics
# ══════════════════════════════════════════════
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)

    if len(st.session_state.history) == 0:
        st.markdown("""<div style="text-align:center;padding:4rem;color:#2a2a40;">
            <div style="font-size:56px;margin-bottom:16px;filter:grayscale(1) opacity(0.2)">📊</div>
            <p style="font-size:16px;font-weight:600;color:#3a3a58">No data yet</p>
            <p style="font-size:13px;margin-top:8px">Run the live detection first to see analytics</p>
        </div>""", unsafe_allow_html=True)
    else:
        df = pd.DataFrame(list(st.session_state.history))
        counts = df['emotion'].value_counts()

        # Metrics
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("🎬 Frames Analyzed", st.session_state.total_frames)
        m2.metric("🎭 Dominant Emotion", counts.index[0].capitalize() if len(counts)>0 else "—")
        m3.metric("😊 Happy Frames", int(counts.get('happy',0)))
        m4.metric("😠 Angry Frames", int(counts.get('angry',0)))

        st.markdown("<br>", unsafe_allow_html=True)
        ch1, ch2 = st.columns(2)

        with ch1:
            fig, ax = plt.subplots(figsize=(5,4.5))
            fig.patch.set_facecolor('#0d0d18'); ax.set_facecolor('#0d0d18')
            cols_pie = [EMOTIONS.get(e,{}).get('color','#6b7280') for e in counts.index]
            wedges,texts,autos = ax.pie(counts.values, labels=[EMOTIONS.get(e,{}).get('label',e) for e in counts.index],
                colors=cols_pie, autopct='%1.1f%%', startangle=140, pctdistance=0.72,
                wedgeprops=dict(width=0.55, edgecolor='#0d0d18', linewidth=3))
            for t in texts: t.set_color('#9ca3af'); t.set_fontsize(11)
            for a in autos: a.set_color('white'); a.set_fontsize(10); a.set_fontweight('700')
            ax.set_title("Emotion Distribution", color='#e2e2ef', fontsize=14, fontweight='800', pad=16)
            st.pyplot(fig); plt.close()

        with ch2:
            emo_cols = [e for e in EMOTIONS.keys() if e in df.columns]
            if emo_cols:
                fig, ax = plt.subplots(figsize=(5,4.5))
                fig.patch.set_facecolor('#0d0d18'); ax.set_facecolor('#0d0d18')
                for emo in emo_cols:
                    color = EMOTIONS[emo]['color']
                    ax.plot(range(len(df)), df[emo], label=EMOTIONS[emo]['label'],
                            color=color, linewidth=1.5, alpha=0.85)
                ax.set_facecolor('#0d0d18')
                ax.tick_params(colors='#6b7280', labelsize=9)
                ax.set_title("Emotion Timeline", color='#e2e2ef', fontsize=14, fontweight='800', pad=16)
                ax.legend(facecolor='#1a1a2e', edgecolor='none', labelcolor='white', fontsize=9, ncol=2)
                ax.yaxis.grid(True, color='#1a1a2e', linewidth=0.6)
                for sp in ax.spines.values(): sp.set_visible(False)
                ax.set_xlabel("Frame", color='#4b5563', fontsize=9)
                ax.set_ylabel("Confidence %", color='#4b5563', fontsize=9)
                st.pyplot(fig); plt.close()

        st.markdown("<br>**📋 Detection Log**")
        st.dataframe(df[['time','emotion']].rename(columns={'time':'Time','emotion':'Detected Emotion'}),
                     use_container_width=True, height=250)
        st.download_button("⬇️ Export Session Data", df.to_csv(index=False),
                           "emotion_session.csv", "text/csv")

# ══════════════════════════════════════════════
# TAB 3 — Screenshots
# ══════════════════════════════════════════════
with tab3:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("Screenshots are captured automatically during detection sessions.")
    st.markdown("""<div style="text-align:center;padding:3rem;color:#2a2a40;
        background:rgba(255,255,255,0.02);border:2px dashed rgba(255,255,255,0.06);border-radius:16px;">
        <div style="font-size:48px;margin-bottom:12px;filter:opacity(0.2)">📸</div>
        <p style="font-size:15px;font-weight:600;color:#3a3a58">Screenshot feature</p>
        <p style="font-size:13px;margin-top:6px;color:#252540">Use your system screenshot tool (Win+Shift+S)<br>while the live detection is running</p>
    </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <strong>EmotionAI</strong> — Built by
    <a href="https://github.com/MuhammadZafran33" target="_blank">Muhammad Zafran</a> ·
    Powered by DeepFace & OpenCV ·
    <a href="https://fiverr.com/muh_zafran" target="_blank">Hire me on Fiverr</a>
</div>
""", unsafe_allow_html=True)
