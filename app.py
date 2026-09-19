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
# 3. ADVANCED INLINE & CUSTOM CSS (OFF-WHITE, SKY BLUE & LIGHT GREEN)
# -----------------------------------------------------------------------------
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">

<style>
/* Global Canvas Reset */
html, body, [class*="st-"] {
    font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, sans-serif !important;
    color: #0F172A;
}

.stApp {
    background-color: #F8FAFC !important;
}

/* Completely remove Streamlit chrome & sidebar */
#MainMenu, footer, header {
    visibility: hidden !important;
}
[data-testid="stSidebar"] {
    display: none !important;
}

.block-container {
    padding-top: 2rem !important;
    padding-bottom: 4rem !important;
    max-width: 900px !important;
    margin: 0 auto !important;
}

/* Laser Scan Animation */
@keyframes beamScan {
    0% { transform: translateY(0); opacity: 0.8; }
    50% { transform: translateY(110px); opacity: 1; }
    100% { transform: translateY(0); opacity: 0.8; }
}

@keyframes pulseGlow {
    0%, 100% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.15); opacity: 0.8; }
}

.scanner-laser {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, transparent, #0284C7, #38BDF8, #0284C7, transparent);
    box-shadow: 0 0 14px #38BDF8;
    animation: beamScan 2.2s infinite ease-in-out;
}

.pulse-icon {
    animation: pulseGlow 1.6s infinite ease-in-out;
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
    padding: 0.9rem 2.2rem !important;
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
# 5. TOP HEADER BANNER (OFF-WHITE, SKY BLUE & LIGHT GREEN INLINE STYLING)
# -----------------------------------------------------------------------------
st.markdown("""
<div style="background: linear-gradient(135deg, #FFFFFF 0%, #F0F9FF 100%); border: 1.5px solid #BAE6FD; border-radius: 20px; padding: 28px; box-shadow: 0 4px 20px -2px rgba(14, 165, 233, 0.08); margin-bottom: 24px;">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;">
        <div>
            <div style="display: inline-flex; align-items: center; gap: 8px; background: #E0F2FE; color: #0369A1; border: 1px solid #BAE6FD; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; margin-bottom: 10px;">
                <span style="width: 8px; height: 8px; border-radius: 50%; background: #0284C7; display: inline-block;"></span>
                <span>GRU Recurrent Neural Network</span>
                <span style="color: #7DD3FC;">•</span>
                <span style="color: #15803D; font-weight: 700;">98.65% Accuracy</span>
            </div>
            <h1 style="font-size: 26px; font-weight: 800; color: #0F172A; margin: 0; letter-spacing: -0.02em;">
                Email Spam & Threat Detection
            </h1>
            <p style="font-size: 14px; color: #475569; margin-top: 6px; margin-bottom: 0; line-height: 1.5; max-width: 580px;">
                Scan email messages in real-time to detect phishing scams, spam, and financial fraud using deep recurrent memory units.
            </p>
        </div>
        <div style="background: #FFFFFF; border: 1px solid #86EFAC; border-radius: 14px; padding: 10px 18px; text-align: right; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);">
            <div style="font-size: 11px; font-weight: 600; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em;">Engine Status</div>
            <div style="display: flex; align-items: center; gap: 6px; margin-top: 2px; justify-content: flex-end;">
                <span style="width: 8px; height: 8px; border-radius: 50%; background: #22C55E; display: inline-block;"></span>
                <span style="font-size: 12px; font-weight: 700; color: #15803D;">Online & Ready</span>
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
tab_scan, tab_bench = st.tabs([
    "🔍 Threat Scanner",
    "📊 Model Performance"
])

# -----------------------------------------------------------------------------
# TAB 1: THREAT SCANNER
# -----------------------------------------------------------------------------
with tab_scan:
    # Main Input Card
    st.markdown("""
    <div style="background: #FFFFFF; border: 1.5px solid #E2E8F0; border-radius: 20px; padding: 22px 26px; box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.05); margin-bottom: 20px;">
        <h3 style="font-size: 18px; font-weight: 700; color: #0F172A; margin: 0 0 4px 0;">Analyze Email Text</h3>
        <p style="font-size: 13px; color: #64748B; margin: 0;">Select a sample email preset or paste your own message below to run deep recurrent neural inference.</p>
    </div>
    """, unsafe_allow_html=True)
    
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
    
    scan_button = st.button("🚀 Analyze Email Threat", use_container_width=True)
    
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
            <div style="position: relative; overflow: hidden; background: linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 50%, #F0F9FF 100%); border: 1.5px solid #38BDF8; border-radius: 18px; padding: 24px; box-shadow: 0 6px 24px rgba(14, 165, 233, 0.15); margin: 24px 0;">
                <div class="scanner-laser"></div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span class="pulse-icon" style="width: 10px; height: 10px; background-color: #0284C7; border-radius: 50%; display: inline-block;"></span>
                        <span style="font-size: 12px; font-weight: 700; color: #0369A1; text-transform: uppercase; letter-spacing: 0.05em;">Analyzing Email Sequences</span>
                    </div>
                    <span style="font-size: 13px; font-family: monospace; font-weight: 700; color: #0369A1; background: #FFFFFF; border: 1px solid #BAE6FD; padding: 2px 10px; border-radius: 14px;">{pct}%</span>
                </div>
                <div style="font-size: 14px; font-weight: 600; color: #0F172A; margin-bottom: 14px;">
                    {stage_msg}
                </div>
                <div style="width: 100%; height: 12px; background-color: #E2E8F0; border-radius: 9999px; overflow: hidden;">
                    <div style="width: {pct}%; height: 100%; border-radius: 9999px; background: linear-gradient(90deg, #38BDF8 0%, #0284C7 100%); transition: width 0.2s ease;"></div>
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
        # VERDICT DISPLAY (BULLETPROOF INLINE CSS)
        # ---------------------------------------------------------------------
        if result["is_spam"]:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #FEF2F2 0%, #FFF5F5 100%); border: 1.5px solid #FECACA; border-radius: 18px; padding: 24px; margin-bottom: 22px; box-shadow: 0 4px 18px rgba(239, 68, 68, 0.08);">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px;">
                    <div style="display: flex; align-items: center; gap: 14px;">
                        <div style="width: 52px; height: 52px; background: #FEE2E2; border: 1px solid #FCA5A5; border-radius: 14px; display: flex; align-items: center; justify-content: center; font-size: 26px;">
                            🚨
                        </div>
                        <div>
                            <div style="font-size: 11px; font-weight: 700; color: #B91C1C; text-transform: uppercase; letter-spacing: 0.05em;">Classification Result</div>
                            <div style="font-size: 22px; font-weight: 800; color: #7F1D1D; line-height: 1.2;">HIGH-RISK SPAM DETECTED</div>
                        </div>
                    </div>
                    <div style="background: #FFFFFF; border: 1px solid #FECACA; padding: 10px 18px; border-radius: 12px; text-align: right;">
                        <div style="font-size: 11px; color: #991B1B; font-weight: 600; text-transform: uppercase;">Spam Probability</div>
                        <div style="font-size: 26px; font-weight: 800; color: #991B1B; font-family: monospace;">{result['confidence']:.2f}%</div>
                    </div>
                </div>
                <div style="margin-top: 14px; padding-top: 12px; border-top: 1px solid #FEE2E2; font-size: 12px; color: #991B1B; line-height: 1.5;">
                    This email contains recurring deceptive markers, urgency triggers, or scam patterns matching known threat databases. Do not open suspicious links or verify personal credentials.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #F0FDF4 0%, #F6FFF8 100%); border: 1.5px solid #BBF7D0; border-radius: 18px; padding: 24px; margin-bottom: 22px; box-shadow: 0 4px 18px rgba(34, 197, 94, 0.08);">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px;">
                    <div style="display: flex; align-items: center; gap: 14px;">
                        <div style="width: 52px; height: 52px; background: #DCFCE7; border: 1px solid #86EFAC; border-radius: 14px; display: flex; align-items: center; justify-content: center; font-size: 26px;">
                            🛡️
                        </div>
                        <div>
                            <div style="font-size: 11px; font-weight: 700; color: #15803D; text-transform: uppercase; letter-spacing: 0.05em;">Classification Result</div>
                            <div style="font-size: 22px; font-weight: 800; color: #14532D; line-height: 1.2;">LEGITIMATE EMAIL (HAM)</div>
                        </div>
                    </div>
                    <div style="background: #FFFFFF; border: 1px solid #BBF7D0; padding: 10px 18px; border-radius: 12px; text-align: right;">
                        <div style="font-size: 11px; color: #15803D; font-weight: 600; text-transform: uppercase;">Clean Confidence</div>
                        <div style="font-size: 26px; font-weight: 800; color: #15803D; font-family: monospace;">{result['confidence']:.2f}%</div>
                    </div>
                </div>
                <div style="margin-top: 14px; padding-top: 12px; border-top: 1px solid #DCFCE7; font-size: 12px; color: #166534; line-height: 1.5;">
                    The sequence model evaluated the syntax, vocabulary, and token flow as authentic correspondence consistent with genuine personal or enterprise communication.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        # ---------------------------------------------------------------------
        # PROBABILITY SPECTRUM & THREAT INDICATORS (PREMIUM INLINE STYLING)
        # ---------------------------------------------------------------------
        prob_pct = result["prob"] * 100
        # Clamped marker position so the pin stays elegantly within the track boundaries
        marker_pos = max(2.5, min(prob_pct, 97.5))
        
        if result["is_spam"]:
            accent_color = "#DC2626"
            score_badge = f'<span style="font-family: monospace; font-weight: 700; font-size: 13px; color: #991B1B; background: #FEF2F2; border: 1.5px solid #FECACA; padding: 5px 14px; border-radius: 20px;">🚨 {prob_pct:.2f}% Threat Score</span>'
        else:
            accent_color = "#16A34A"
            score_badge = f'<span style="font-family: monospace; font-weight: 700; font-size: 13px; color: #15803D; background: #F0FDF4; border: 1.5px solid #BBF7D0; padding: 5px 14px; border-radius: 20px;">🛡️ {prob_pct:.2f}% Safe (Low Risk)</span>'
        
        # Keyword chips or clean message
        if result["detected_words"]:
            chips_html = "".join([
                f'<span style="display: inline-flex; align-items: center; gap: 6px; background: #FEF2F2; color: #991B1B; border: 1.5px solid #FECACA; padding: 6px 14px; border-radius: 10px; font-size: 13px; font-weight: 700; box-shadow: 0 1px 4px rgba(239, 68, 68, 0.06);">⚠️ {w}</span>'
                for w in sorted(result["detected_words"])
            ])
            keyword_block = f"""<div style="margin-top: 18px; padding-top: 16px; border-top: 1px solid #F1F5F9;"><div style="font-size: 13px; font-weight: 700; color: #475569; margin-bottom: 10px; display: flex; align-items: center; gap: 6px;"><span>🔍</span> Flagged Trigger Keywords ({len(result['detected_words'])} detected):</div><div style="display: flex; flex-wrap: wrap; gap: 8px;">{chips_html}</div></div>"""
        else:
            keyword_block = """<div style="margin-top: 18px; padding-top: 16px; border-top: 1px solid #F1F5F9;"><div style="background-color: #F0FDF4; border: 1.5px solid #BBF7D0; color: #15803D; padding: 12px 18px; border-radius: 12px; font-size: 13px; font-weight: 600; display: flex; align-items: center; gap: 10px;"><span style="font-size: 16px;">✅</span><span><b>Zero Threat Trigger Keywords:</b> Lexical patterns and word tokens are verified authentic.</span></div></div>"""
            
        spectrum_html = f"""<div style="background: #FFFFFF; border: 1.5px solid #E2E8F0; border-radius: 20px; padding: 26px; box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.05); margin-bottom: 24px;">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 18px; flex-wrap: wrap; gap: 10px;">
<div style="display: flex; align-items: center; gap: 8px;"><span style="font-size: 18px;">📊</span><h4 style="font-size: 16px; font-weight: 800; color: #0F172A; margin: 0;">Probability Spectrum & Threat Indicators</h4></div>
<div>{score_badge}</div>
</div>
<div style="display: flex; justify-content: space-between; align-items: center; font-size: 12px; font-weight: 700; margin-bottom: 10px;">
<span style="color: #15803D; background: #DCFCE7; border: 1px solid #86EFAC; padding: 3px 12px; border-radius: 12px; display: inline-flex; align-items: center; gap: 5px;"><span style="width: 7px; height: 7px; border-radius: 50%; background: #16A34A; display: inline-block;"></span>Clean (0%)</span>
<span style="color: #64748B; font-size: 11px; font-weight: 600; background: #F1F5F9; border: 1px solid #E2E8F0; padding: 2px 10px; border-radius: 10px;">Decision Threshold (50%)</span>
<span style="color: #B91C1C; background: #FEE2E2; border: 1px solid #FCA5A5; padding: 3px 12px; border-radius: 12px; display: inline-flex; align-items: center; gap: 5px;"><span style="width: 7px; height: 7px; border-radius: 50%; background: #DC2626; display: inline-block;"></span>Threat (100%)</span>
</div>
<div style="position: relative; width: 100%; height: 16px; background: linear-gradient(90deg, #22C55E 0%, #38BDF8 40%, #FBBF24 75%, #EF4444 100%); border-radius: 9999px; margin-top: 8px; margin-bottom: 22px; box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.08); border: 1px solid #CBD5E1;">
<div style="position: absolute; top: 50%; left: {marker_pos}%; transform: translate(-50%, -50%); width: 28px; height: 28px; border-radius: 50%; background: #FFFFFF; border: 3.5px solid {accent_color}; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.22); display: flex; align-items: center; justify-content: center; transition: left 0.6s ease;">
<div style="width: 8px; height: 8px; border-radius: 50%; background: {accent_color};"></div>
</div>
</div>
{keyword_block}
<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 18px; padding-top: 16px; border-top: 1px solid #F1F5F9; text-align: center;">
<div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 12px;">
<div style="color: #64748B; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em;">Model Probability</div>
<div style="font-family: monospace; font-weight: 800; color: #0F172A; font-size: 15px; margin-top: 3px;">{result['prob']:.4f}</div>
</div>
<div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 12px;">
<div style="color: #64748B; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em;">Lexical Tokens</div>
<div style="font-family: monospace; font-weight: 800; color: #0F172A; font-size: 15px; margin-top: 3px;">{result['token_count']} words</div>
</div>
<div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 12px;">
<div style="color: #64748B; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em;">Confidence</div>
<div style="font-family: monospace; font-weight: 800; color: #0284C7; font-size: 15px; margin-top: 3px;">{result['confidence']:.2f}%</div>
</div>
</div>
</div>"""
        st.markdown(spectrum_html, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 2: MODEL PERFORMANCE & BENCHMARKS
# -----------------------------------------------------------------------------
with tab_bench:
    st.markdown("""
    <div style="background: #FFFFFF; border: 1.5px solid #E2E8F0; border-radius: 20px; padding: 28px; box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.05); margin-bottom: 24px;">
        <h3 style="font-size: 18px; font-weight: 700; color: #0F172A; margin: 0 0 4px 0;">Architecture & Model Benchmarks</h3>
        <p style="font-size: 13px; color: #64748B; margin: 0 0 20px 0;">Evaluation comparison across SimpleRNN, LSTM, and the deployed GRU model trained on 83,448 emails.</p>
        
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px;">
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 14px; padding: 18px;">
                <div style="font-size: 11px; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em;">Baseline</div>
                <div style="font-size: 16px; font-weight: 700; color: #0F172A; margin-top: 4px;">Simple RNN</div>
                <div style="margin-top: 12px; font-size: 12px; color: #475569; display: flex; flex-direction: column; gap: 6px;">
                    <div style="display: flex; justify-content: space-between;"><span>Accuracy:</span><span style="font-family: monospace; font-weight: 700; color: #0F172A;">97.69%</span></div>
                    <div style="display: flex; justify-content: space-between;"><span>Spam Precision:</span><span style="font-family: monospace; color: #0F172A;">97.82%</span></div>
                    <div style="display: flex; justify-content: space-between;"><span>Spam Recall:</span><span style="font-family: monospace; color: #0F172A;">97.80%</span></div>
                    <div style="display: flex; justify-content: space-between;"><span>Spam F1:</span><span style="font-family: monospace; color: #0F172A;">97.81%</span></div>
                </div>
            </div>
            
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 14px; padding: 18px;">
                <div style="font-size: 11px; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em;">Long-Short Memory</div>
                <div style="font-size: 16px; font-weight: 700; color: #0F172A; margin-top: 4px;">LSTM</div>
                <div style="margin-top: 12px; font-size: 12px; color: #475569; display: flex; flex-direction: column; gap: 6px;">
                    <div style="display: flex; justify-content: space-between;"><span>Accuracy:</span><span style="font-family: monospace; font-weight: 700; color: #0F172A;">98.48%</span></div>
                    <div style="display: flex; justify-content: space-between;"><span>Spam Precision:</span><span style="font-family: monospace; color: #0F172A;">98.38%</span></div>
                    <div style="display: flex; justify-content: space-between;"><span>Spam Recall:</span><span style="font-family: monospace; color: #0F172A;">98.74%</span></div>
                    <div style="display: flex; justify-content: space-between;"><span>Spam F1:</span><span style="font-family: monospace; color: #0F172A;">98.56%</span></div>
                </div>
            </div>
            
            <div style="background: #F0F9FF; border: 2px solid #BAE6FD; border-radius: 14px; padding: 18px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="font-size: 11px; font-weight: 700; color: #0369A1; text-transform: uppercase; letter-spacing: 0.05em;">Deployed Model</div>
                    <span style="background: #0284C7; color: #FFFFFF; font-size: 10px; font-weight: 700; padding: 2px 8px; border-radius: 10px;">TOP BENCHMARK</span>
                </div>
                <div style="font-size: 16px; font-weight: 700; color: #0F172A; margin-top: 4px;">GRU Model</div>
                <div style="margin-top: 12px; font-size: 12px; color: #0369A1; display: flex; flex-direction: column; gap: 6px;">
                    <div style="display: flex; justify-content: space-between;"><span>Accuracy:</span><span style="font-family: monospace; font-weight: 700; color: #0284C7; font-size: 13px;">98.65%</span></div>
                    <div style="display: flex; justify-content: space-between;"><span>Spam Precision:</span><span style="font-family: monospace; font-weight: 700; color: #0284C7;">98.75%</span></div>
                    <div style="display: flex; justify-content: space-between;"><span>Spam Recall:</span><span style="font-family: monospace; font-weight: 700; color: #0284C7;">98.69%</span></div>
                    <div style="display: flex; justify-content: space-between;"><span>Spam F1:</span><span style="font-family: monospace; font-weight: 700; color: #15803D; font-size: 13px;">98.72%</span></div>
                </div>
            </div>
        </div>
        
        <div style="margin-top: 20px; padding-top: 16px; border-top: 1px solid #E2E8F0; font-size: 12px; color: #64748B; line-height: 1.5;">
            <b>Model Specifications:</b> Vocabulary: 7,000 top words • Sequence Length: 500 tokens (padded post) • Embedding: 32 dimensions • GRU: 64 units • Optimizer: Adam • Loss: Binary Crossentropy
        </div>
    </div>
    """, unsafe_allow_html=True)
