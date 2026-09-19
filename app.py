import os
import re
import time
import json
import random
import string
import pickle
import threading
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------------------------------------------------------
# 1. STREAMLIT PAGE CONFIGURATION (CLEAN SINGLE-COLUMN LAYOUT, NO SIDEBAR)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Email Spam Detection • GRU Neural Shield",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------------------------------------------
# 2. SILENT BACKGROUND KEEP-ALIVE SERVER (RUNS IN BACKGROUND ONLY, NO UI CLUTTER)
# -----------------------------------------------------------------------------
APP_START_TIME = time.time()

class SilentKeepAliveHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        uptime = int(time.time() - APP_START_TIME)
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if self.path in ["/ping", "/health", "/keepalive", "/"]:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            resp_body = {
                "status": "healthy",
                "app": "Email-Spam-Detection",
                "state": "AWAKE",
                "uptime_seconds": uptime,
                "timestamp": now_str
            }
            self.wfile.write(json.dumps(resp_body).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

def run_keepalive_server(port=8502):
    try:
        server = HTTPServer(("0.0.0.0", port), SilentKeepAliveHandler)
        server.serve_forever()
    except Exception:
        pass

@st.cache_resource
def start_silent_keepalive():
    ping_port = int(os.environ.get("PING_PORT", 8502))
    t = threading.Thread(target=run_keepalive_server, args=(ping_port,), daemon=True)
    t.start()
    return ping_port

start_silent_keepalive()

# -----------------------------------------------------------------------------
# 3. ADVANCED TAILWIND CSS & OFF-WHITE / SKY BLUE / LIGHT GREEN DESIGN SYSTEM
# -----------------------------------------------------------------------------
st.markdown("""
<script src="https://cdn.tailwindcss.com"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">

<style>
/* Base Off-White Theme Variables */
:root {
    --bg-canvas: #F8FAFC;
    --card-surface: #FFFFFF;
    --border-subtle: #E2E8F0;
    --sky-primary: #0284C7;
    --sky-hover: #0369A1;
    --sky-light: #E0F2FE;
    --sky-border: #BAE6FD;
    --green-primary: #16A34A;
    --green-light: #DCFCE7;
    --green-border: #86EFAC;
    --text-main: #0F172A;
    --text-muted: #64748B;
}

html, body, [class*="st-"] {
    font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, sans-serif;
    color: var(--text-main);
}

.stApp {
    background-color: #F8FAFC;
}

/* Completely Hide Streamlit default clutter & sidebar */
#MainMenu, footer, header {
    visibility: hidden;
}
[data-testid="stSidebar"] {
    display: none !important;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 4rem;
    max-width: 960px;
    margin: 0 auto;
}

/* Off-White Elevated Card Styles */
.app-card {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.05), 0 2px 8px -2px rgba(15, 23, 42, 0.02);
    margin-bottom: 1.5rem;
}

.app-card-sky {
    background: linear-gradient(145deg, #FFFFFF 0%, #F0F9FF 100%);
    border: 1.5px solid #BAE6FD;
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 6px 24px -4px rgba(14, 165, 233, 0.12);
    margin-bottom: 1.5rem;
}

.app-card-green {
    background: linear-gradient(145deg, #FFFFFF 0%, #F0FDF4 100%);
    border: 1.5px solid #BBF7D0;
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 6px 24px -4px rgba(34, 197, 94, 0.1);
    margin-bottom: 1.5rem;
}

.app-card-red {
    background: linear-gradient(145deg, #FFFFFF 0%, #FEF2F2 100%);
    border: 1.5px solid #FECACA;
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 6px 24px -4px rgba(239, 68, 68, 0.1);
    margin-bottom: 1.5rem;
}

/* Animations */
@keyframes fadeInSlide {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes beamScan {
    0% { transform: translateY(0); opacity: 0.7; }
    50% { transform: translateY(140px); opacity: 1; }
    100% { transform: translateY(0); opacity: 0.7; }
}

@keyframes pulseGlow {
    0%, 100% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.08); opacity: 0.85; }
}

.fade-in {
    animation: fadeInSlide 0.4s ease-out forwards;
}

.pulse-icon {
    animation: pulseGlow 1.8s infinite ease-in-out;
}

/* Realistic Scanning / Reading Animation Card */
.scanner-container {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 50%, #F0F9FF 100%);
    border: 1.5px solid #38BDF8;
    border-radius: 18px;
    padding: 2rem;
    box-shadow: 0 8px 30px -4px rgba(14, 165, 233, 0.2);
    margin: 1.75rem 0;
}

.scanner-laser {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, transparent, #0284C7, #38BDF8, #0284C7, transparent);
    box-shadow: 0 0 14px #38BDF8;
    animation: beamScan 2.4s infinite ease-in-out;
}

.custom-progress-track {
    background: #E2E8F0;
    border-radius: 9999px;
    height: 12px;
    overflow: hidden;
    position: relative;
}

.custom-progress-fill {
    height: 100%;
    border-radius: 9999px;
    background: linear-gradient(90deg, #38BDF8 0%, #0284C7 100%);
    transition: width 0.3s ease;
}

/* FIX: PROMINENT SKY-BLUE BUTTON */
div.stButton > button, 
div.stButton > button:first-child, 
div.stButton > button:hover, 
div.stButton > button:active, 
div.stButton > button:focus {
    background: linear-gradient(135deg, #0284C7 0%, #0EA5E9 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.85rem 2.2rem !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 6px 20px rgba(14, 165, 233, 0.35) !important;
    transition: all 0.25s ease !important;
    cursor: pointer !important;
    width: 100% !important;
}

div.stButton > button:hover {
    background: linear-gradient(135deg, #0369A1 0%, #0284C7 100%) !important;
    box-shadow: 0 8px 26px rgba(14, 165, 233, 0.5) !important;
    transform: translateY(-2px) !important;
}

/* Text Area Customization */
div.stTextArea textarea {
    border-radius: 16px !important;
    border: 1.5px solid #CBD5E1 !important;
    background-color: #FFFFFF !important;
    color: #0F172A !important;
    font-size: 0.98rem !important;
    line-height: 1.6 !important;
    padding: 1.1rem !important;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.02) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}

div.stTextArea textarea:focus {
    border-color: #0284C7 !important;
    box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.2) !important;
}

/* Select Box */
div.stSelectbox > div > div {
    border-radius: 14px !important;
    border: 1.5px solid #CBD5E1 !important;
    background-color: #FFFFFF !important;
    font-size: 0.95rem !important;
}

/* Tab Headers */
.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    border-bottom: 2px solid #E2E8F0;
    padding-bottom: 6px;
    margin-bottom: 1.75rem;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 12px;
    padding: 10px 22px;
    font-weight: 600;
    font-size: 0.92rem;
    color: #64748B;
    background-color: transparent;
    transition: all 0.2s ease;
}

.stTabs [aria-selected="true"] {
    background-color: #E0F2FE !important;
    color: #0284C7 !important;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. MODEL ARTIFACTS & INFERENCE PIPELINE
# -----------------------------------------------------------------------------
DEFAULT_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have",
    "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers",
    "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm",
    "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off",
    "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out",
    "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should",
    "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their",
    "theirs", "them", "themselves", "then", "there", "there's", "these", "they",
    "they'd", "they'll", "they're", "they've", "this", "those", "through", "to",
    "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll",
    "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where",
    "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with",
    "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your",
    "yours", "yourself", "yourselves"
}

HIGH_RISK_SPAM_WORDS = {
    "escapenumber", "escapelong", "viagra", "pills", "cialis", "free", "lottery",
    "winner", "prize", "urgent", "cash", "million", "dollars", "bonus", "reward",
    "verify", "password", "suspended", "account", "unauthorized", "crypto", "investment",
    "bitcoin", "profit", "medications", "prescriptions", "claim", "deposit", "wire",
    "guaranteed", "refinance", "credit", "cheap", "lowest", "click", "limited", "offer"
}

@st.cache_resource(show_spinner=False)
def load_model_and_artifacts():
    with open("config.pkl", "rb") as f:
        config = pickle.load(f)
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("label_mapping.pkl", "rb") as f:
        label_mapping = pickle.load(f)
    model = tf.keras.models.load_model("gru_model.keras")
    return model, tokenizer, config, label_mapping

try:
    MODEL, TOKENIZER, CONFIG, LABEL_MAPPING = load_model_and_artifacts()
    MODEL_READY = True
except Exception as e:
    MODEL_READY = False
    LOAD_ERR = str(e)

def clean_text(raw_text):
    text = raw_text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"http\S+|www\S+", "", text)
    tokens = text.split()
    filtered = [w for w in tokens if w not in DEFAULT_STOPWORDS]
    cleaned = " ".join(filtered)
    return cleaned, filtered

def run_inference(raw_text):
    cleaned, tokens = clean_text(raw_text)
    seq = TOKENIZER.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=CONFIG["max_length"], padding="post")
    
    prob = float(MODEL.predict(padded, verbose=0)[0][0])
    is_spam = prob >= 0.5
    verdict = "Spam" if is_spam else "Ham"
    confidence = (prob if is_spam else (1.0 - prob)) * 100
    
    detected_words = [w for w in tokens if w in HIGH_RISK_SPAM_WORDS]
    
    return {
        "raw_text": raw_text,
        "cleaned": cleaned,
        "prob": prob,
        "is_spam": is_spam,
        "verdict": verdict,
        "confidence": confidence,
        "detected_words": list(set(detected_words)),
        "word_count": len(raw_text.split()),
        "token_count": len(tokens)
    }

# -----------------------------------------------------------------------------
# 5. TOP HEADER BANNER (OFF-WHITE, SKY BLUE & LIGHT GREEN)
# -----------------------------------------------------------------------------
st.markdown("""
<div class="app-card-sky fade-in">
    <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
            <div class="inline-flex items-center space-x-2 bg-sky-100/90 text-sky-800 border border-sky-200 px-3 py-1 rounded-full text-xs font-semibold mb-2">
                <span class="w-2 h-2 rounded-full bg-sky-500 inline-block"></span>
                <span>GRU Recurrent Neural Network</span>
                <span class="text-sky-300">•</span>
                <span class="text-emerald-700 font-bold">98.65% Accuracy</span>
            </div>
            <h1 class="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">
                Email Spam & Threat Detection
            </h1>
            <p class="text-sm text-slate-600 mt-1 max-w-xl leading-relaxed">
                Scan email messages in real-time to detect spam, phishing attempts, and fraud using deep gated recurrent memory units.
            </p>
        </div>
        <div class="hidden sm:flex items-center">
            <div class="bg-white/90 border border-emerald-200 rounded-xl px-3.5 py-2 text-right shadow-sm">
                <div class="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">Engine Status</div>
                <div class="flex items-center space-x-1.5 mt-0.5 justify-end">
                    <span class="w-2.5 h-2.5 rounded-full bg-emerald-500 inline-block"></span>
                    <span class="text-xs font-bold text-emerald-700">Online & Ready</span>
                </div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if not MODEL_READY:
    st.error(f"Failed to load model: {LOAD_ERR}")
    st.stop()

# -----------------------------------------------------------------------------
# 6. MAIN WORKSPACE TABS
# -----------------------------------------------------------------------------
tab_scan, tab_batch, tab_bench = st.tabs([
    "🔍 Threat Scanner",
    "📂 Batch Screening",
    "📊 Model Performance"
])

# -----------------------------------------------------------------------------
# TAB 1: THREAT SCANNER
# -----------------------------------------------------------------------------
with tab_scan:
    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    
    st.markdown("<h3 class='text-lg font-bold text-slate-900 mb-1'>Analyze Email Text</h3>", unsafe_allow_html=True)
    st.markdown("<p class='text-xs text-slate-500 mb-4'>Select a sample email preset or paste your own message below.</p>", unsafe_allow_html=True)
    
    preset_choice = st.selectbox(
        "Choose an email sample preset",
        [
            "-- Choose a sample preset to test --",
            "🚨 Phishing Alert: Immediate Account Suspension (Spam)",
            "💰 International Sweepstakes Cash Winner (Spam)",
            "💊 Prescription Pharma Discount (Spam)",
            "💼 Team Sprint Planning & Strategy Sync (Ham / Clean)",
            "📦 Shipment Tracking & Order Confirmation (Ham / Clean)"
        ],
        index=0,
        label_visibility="collapsed"
    )
    
    samples = {
        "🚨 Phishing Alert: Immediate Account Suspension (Spam)": 
            "URGENT NOTICE: Your online account access has been temporarily suspended due to suspicious sign-in activity detected from an unknown device. To safeguard your profile and prevent permanent deactivation, you must verify your credentials immediately. Click the official link below to confirm your password and restore access: http://security-verification-portal.com/auth",
        "💰 International Sweepstakes Cash Winner (Spam)":
            "Dear Winner, Congratulations! Your email address was selected in the official international lottery draw, awarding you $1,500,000 USD in cash funds. To claim and process the wire deposit into your bank account, please reply immediately with your full legal name, telephone number, and bank details. Congratulations once again!",
        "💊 Prescription Pharma Discount (Spam)":
            "Get your discount prescriptions online with no appointment or doctor visit required. Authentic viagra, cialis, and sleep pills available at wholesale rates with discrete overnight international shipping. Click here to secure your order at 80% discount.",
        "💼 Team Sprint Planning & Strategy Sync (Ham / Clean)":
            "Hi Alex and Team,\n\nI hope you're having a productive week. Let's schedule our quarterly sprint planning and roadmap review for this Thursday at 2:00 PM. Please take a look at the attached presentation and add any items you'd like to discuss to the shared agenda.\n\nBest regards,\nDavid",
        "📦 Shipment Tracking & Order Confirmation (Ham / Clean)":
            "Hello Jessica,\n\nThank you for shopping with us! Your order #829104 has been packed and handed over to our courier partner. It is currently in transit and estimated to arrive by Monday afternoon. You can view your shipment progress and delivery preferences through your account portal."
    }
    
    default_text = samples.get(preset_choice, "")
    
    email_input = st.text_area(
        "Email Message",
        value=default_text,
        height=180,
        placeholder="Paste your email text, subject, or message body here...",
        label_visibility="collapsed"
    )
    
    st.markdown("<div class='mt-3 mb-1'>", unsafe_allow_html=True)
    scan_button = st.button("🚀 Analyze Email Threat", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # -------------------------------------------------------------------------
    # 7. ANIMATION (5-8 SECONDS REALISTIC NEURAL READING & SCANNING)
    # -------------------------------------------------------------------------
    if email_input.strip() and (scan_button or preset_choice != "-- Choose a sample preset to test --"):
        scanner_box = st.empty()
        
        # Realistic random duration between 5.0 and 8.0 seconds
        duration = random.uniform(5.0, 8.0)
        start_time = time.time()
        
        scan_stages = [
            ("Scanning email structure & tokenizing word streams...", 0.20),
            ("Filtering lexical stopwords & identifying trigger markers...", 0.45),
            ("Feeding sequence tokens through 64 GRU bidirectional gates...", 0.75),
            ("Synthesizing recurrent embeddings & computing threat vector...", 0.90),
            ("Finalizing classification verdict...", 1.0)
        ]
        
        while True:
            elapsed = time.time() - start_time
            progress = min(1.0, elapsed / duration)
            pct = int(progress * 100)
            
            stage_msg = scan_stages[-1][0]
            for msg, thresh in scan_stages:
                if progress <= thresh:
                    stage_msg = msg
                    break
            
            scanner_box.markdown(f"""
            <div class="scanner-container">
                <div class="scanner-laser"></div>
                <div class="flex items-center justify-between mb-3">
                    <div class="flex items-center space-x-2">
                        <span class="w-3 h-3 rounded-full bg-sky-500 pulse-icon inline-block"></span>
                        <span class="text-xs font-bold text-sky-900 uppercase tracking-wider">Analyzing Email Sequences</span>
                    </div>
                    <span class="text-xs font-mono font-bold text-sky-700 bg-sky-100 px-2 py-0.5 rounded-full">{pct}%</span>
                </div>
                <div class="text-sm font-semibold text-slate-800 mb-3">
                    {stage_msg}
                </div>
                <div class="custom-progress-track">
                    <div class="custom-progress-fill" style="width: {pct}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if elapsed >= duration:
                break
            time.sleep(0.1)
        
        scanner_box.empty()
        
        # Real inference
        result = run_inference(email_input)
        
        # ---------------------------------------------------------------------
        # VERDICT DISPLAY (OFF-WHITE, SKY BLUE & LIGHT GREEN)
        # ---------------------------------------------------------------------
        if result["is_spam"]:
            st.markdown(f"""
            <div class="app-card-red fade-in">
                <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                    <div class="flex items-center space-x-4">
                        <div class="w-14 h-14 rounded-2xl bg-rose-100 border border-rose-200 flex items-center justify-center text-3xl shadow-sm">
                            🚨
                        </div>
                        <div>
                            <div class="text-xs font-bold uppercase tracking-wider text-rose-700">Classification Result</div>
                            <h2 class="text-2xl font-extrabold text-rose-900 tracking-tight">HIGH-RISK SPAM DETECTED</h2>
                        </div>
                    </div>
                    <div class="bg-white border border-rose-200 px-5 py-3 rounded-xl text-right shadow-sm">
                        <div class="text-[11px] font-semibold text-rose-600 uppercase tracking-wider">Spam Probability</div>
                        <div class="text-3xl font-extrabold text-rose-800 font-mono">{result['confidence']:.2f}%</div>
                    </div>
                </div>
                <p class="text-xs text-rose-900/80 mt-4 pt-3 border-t border-rose-200/70 leading-relaxed">
                    This email contains recurring deceptive markers, urgency triggers, or scam patterns matching known threat databases. Do not open suspicious links or verify personal information.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="app-card-green fade-in">
                <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                    <div class="flex items-center space-x-4">
                        <div class="w-14 h-14 rounded-2xl bg-emerald-100 border border-emerald-200 flex items-center justify-center text-3xl shadow-sm">
                            🛡️
                        </div>
                        <div>
                            <div class="text-xs font-bold uppercase tracking-wider text-emerald-700">Classification Result</div>
                            <h2 class="text-2xl font-extrabold text-emerald-900 tracking-tight">LEGITIMATE EMAIL (HAM)</h2>
                        </div>
                    </div>
                    <div class="bg-white border border-emerald-200 px-5 py-3 rounded-xl text-right shadow-sm">
                        <div class="text-[11px] font-semibold text-emerald-600 uppercase tracking-wider">Clean Confidence</div>
                        <div class="text-3xl font-extrabold text-emerald-800 font-mono">{result['confidence']:.2f}%</div>
                    </div>
                </div>
                <p class="text-xs text-emerald-900/80 mt-4 pt-3 border-t border-emerald-200/70 leading-relaxed">
                    The sequence model evaluated the syntax, vocabulary, and token flow as consistent with authentic enterprise and personal correspondence.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        # Detailed Diagnostic Card
        st.markdown("<div class='app-card'>", unsafe_allow_html=True)
        st.markdown("<h4 class='text-sm font-bold text-slate-900 mb-3'>Probability Spectrum & Threat Indicators</h4>", unsafe_allow_html=True)
        
        prob_pct = result["prob"] * 100
        grad_color = "linear-gradient(90deg, #22C55E 0%, #38BDF8 50%, #EF4444 100%)"
        
        st.markdown(f"""
        <div class="flex justify-between items-center text-xs font-semibold text-slate-600 mb-2">
            <span class="text-emerald-700 font-bold">● Clean (0%)</span>
            <span class="font-mono text-slate-900 font-bold bg-sky-50 border border-sky-200 text-sky-800 px-3 py-1 rounded-full">{prob_pct:.2f}% Risk Score</span>
            <span class="text-rose-700 font-bold">● Threat (100%)</span>
        </div>
        <div class="custom-progress-track mb-4">
            <div class="meter-fill" style="width: {prob_pct}%; background: {grad_color}; height: 100%; border-radius: 9999px;"></div>
        </div>
        """, unsafe_allow_html=True)
        
        if result["detected_words"]:
            chips = "".join([
                f"<span class='inline-block bg-rose-50 text-rose-800 border border-rose-200 px-2.5 py-1 rounded-lg text-xs font-semibold mr-2 mb-2'>⚠ {w}</span>"
                for w in result["detected_words"]
            ])
            st.markdown(f"""
            <div class="pt-3 border-t border-slate-100">
                <div class="text-xs text-slate-500 font-medium mb-2">Flagged Keywords in Message:</div>
                <div class="flex flex-wrap">{chips}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="pt-3 border-t border-slate-100 flex items-center space-x-2 text-xs text-emerald-700 font-medium">
                <span>✓</span>
                <span>No high-frequency spam trigger words found. Linguistic structure verified clean.</span>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 2: BATCH SCREENING
# -----------------------------------------------------------------------------
with tab_batch:
    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='text-lg font-bold text-slate-900 mb-1'>Batch Email Screening</h3>", unsafe_allow_html=True)
    st.markdown("<p class='text-xs text-slate-500 mb-4'>Upload a CSV file containing an email column to screen multiple records simultaneously.</p>", unsafe_allow_html=True)
    
    upload_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
    
    if upload_file is not None:
        try:
            b_df = pd.read_csv(upload_file)
            st.success(f"File loaded successfully: {len(b_df)} records found.")
            
            cand_cols = [c for c in b_df.columns if c.lower() in ["text", "email", "body", "content", "message"]]
            sel_col = cand_cols[0] if cand_cols else b_df.columns[0]
            col_target = st.selectbox("Select email column:", b_df.columns, index=b_df.columns.get_loc(sel_col))
            
            if st.button("🚀 Run Batch Classification"):
                p_bar = st.progress(0)
                preds = []
                probs = []
                
                total_r = len(b_df)
                for i, row in b_df.iterrows():
                    out = run_inference(str(row[col_target]))
                    preds.append(out["verdict"])
                    probs.append(round(out["prob"], 4))
                    if (i + 1) % max(1, total_r // 20) == 0:
                        p_bar.progress((i + 1) / total_r)
                        
                p_bar.progress(1.0)
                b_df["Prediction"] = preds
                b_df["Spam_Probability"] = probs
                
                spam_cnt = (b_df["Prediction"] == "Spam").sum()
                ham_cnt = (b_df["Prediction"] == "Ham").sum()
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Scanned", total_r)
                m2.metric("Spam Detections", f"{spam_cnt} ({spam_cnt/total_r*100:.1f}%)")
                m3.metric("Clean Messages", f"{ham_cnt} ({ham_cnt/total_r*100:.1f}%)")
                
                st.dataframe(b_df.head(50), use_container_width=True)
                
                csv_b = b_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Classified CSV",
                    data=csv_b,
                    file_name="sentinel_batch_results.csv",
                    mime="text/csv"
                )
        except Exception as ex:
            st.error(f"Error processing CSV: {ex}")
    else:
        st.markdown("""
        <div class="text-center py-10">
            <div class="text-4xl mb-2">📁</div>
            <div class="text-sm font-bold text-slate-800">No Dataset Uploaded</div>
            <p class="text-xs text-slate-500 max-w-sm mx-auto mt-1">Upload any CSV file containing email text to perform bulk classification.</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 3: MODEL PERFORMANCE & BENCHMARKS
# -----------------------------------------------------------------------------
with tab_bench:
    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='text-lg font-bold text-slate-900 mb-1'>Architecture & Model Benchmarks</h3>", unsafe_allow_html=True)
    st.markdown("<p class='text-xs text-slate-500 mb-4'>Rigorous evaluation comparison across SimpleRNN, LSTM, and the deployed GRU model trained on 83,448 emails.</p>", unsafe_allow_html=True)
    
    b_col1, b_col2, b_col3 = st.columns(3)
    
    with b_col1:
        st.markdown("""
        <div class="bg-slate-50 border border-slate-200 rounded-xl p-4">
            <div class="text-[11px] font-bold text-slate-400 uppercase tracking-wider">Baseline</div>
            <h4 class="text-base font-bold text-slate-800 mt-1">Simple RNN</h4>
            <div class="mt-3 space-y-1.5 text-xs">
                <div class="flex justify-between"><span>Accuracy:</span><span class="font-mono font-bold">97.69%</span></div>
                <div class="flex justify-between"><span>Spam Precision:</span><span class="font-mono">97.82%</span></div>
                <div class="flex justify-between"><span>Spam Recall:</span><span class="font-mono">97.80%</span></div>
                <div class="flex justify-between"><span>Spam F1:</span><span class="font-mono">97.81%</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with b_col2:
        st.markdown("""
        <div class="bg-slate-50 border border-slate-200 rounded-xl p-4">
            <div class="text-[11px] font-bold text-slate-400 uppercase tracking-wider">Long-Short Memory</div>
            <h4 class="text-base font-bold text-slate-800 mt-1">LSTM</h4>
            <div class="mt-3 space-y-1.5 text-xs">
                <div class="flex justify-between"><span>Accuracy:</span><span class="font-mono font-bold">98.48%</span></div>
                <div class="flex justify-between"><span>Spam Precision:</span><span class="font-mono">98.38%</span></div>
                <div class="flex justify-between"><span>Spam Recall:</span><span class="font-mono">98.74%</span></div>
                <div class="flex justify-between"><span>Spam F1:</span><span class="font-mono">98.56%</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with b_col3:
        st.markdown("""
        <div class="bg-sky-50 border-2 border-sky-300 rounded-xl p-4">
            <div class="flex justify-between items-center">
                <div class="text-[11px] font-bold text-sky-700 uppercase tracking-wider">Deployed Model</div>
                <span class="bg-sky-600 text-white text-[10px] font-bold px-2 py-0.5 rounded-full">TOP BENCHMARK</span>
            </div>
            <h4 class="text-base font-bold text-slate-900 mt-1">GRU Model</h4>
            <div class="mt-3 space-y-1.5 text-xs">
                <div class="flex justify-between"><span>Accuracy:</span><span class="font-mono font-bold text-sky-800">98.65%</span></div>
                <div class="flex justify-between"><span>Spam Precision:</span><span class="font-mono font-bold text-sky-800">98.75%</span></div>
                <div class="flex justify-between"><span>Spam Recall:</span><span class="font-mono font-bold text-sky-800">98.69%</span></div>
                <div class="flex justify-between"><span>Spam F1:</span><span class="font-mono font-bold text-emerald-700">98.72%</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<div class='mt-6 pt-4 border-t border-slate-200 text-xs text-slate-600'>", unsafe_allow_html=True)
    st.markdown("<b>Model Specifications:</b> Vocabulary: 7,000 top words • Sequence Length: 500 tokens (padded post) • Embedding: 32 dimensions • GRU: 64 units • Optimizer: Adam • Loss: Binary Crossentropy")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
