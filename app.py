import os
import re
import time
import json
import string
import pickle
import threading
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
# pyrefly: ignore [missing-import]
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------------------------------------------------------
# 1. PAGE CONFIG & MODERN HEADLESS SETTINGS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Sentinel • Email Spam Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. ANTI-SLEEP KEEP-ALIVE SERVER & SELF-PING DAEMON
# -----------------------------------------------------------------------------
APP_START_TIME = time.time()
PING_STATS = {
    "requests_handled": 0,
    "last_ping": None,
    "self_pings_sent": 0,
    "last_self_ping": None,
    "self_ping_status": "Idle",
    "ping_history": []
}
STATS_LOCK = threading.Lock()

class KeepAliveHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Silence default terminal logs to keep stdout clean
        pass

    def do_GET(self):
        uptime = int(time.time() - APP_START_TIME)
        now_str = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        with STATS_LOCK:
            PING_STATS["requests_handled"] += 1
            PING_STATS["last_ping"] = now_str
            PING_STATS["ping_history"].append({
                "time": now_str,
                "client": self.client_address[0],
                "path": self.path,
                "type": "Inbound Ping"
            })
            if len(PING_STATS["ping_history"]) > 20:
                PING_STATS["ping_history"].pop(0)

        if self.path in ["/ping", "/health", "/keepalive", "/"]:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            response = {
                "status": "healthy",
                "service": "Sentinel Email Spam Detection",
                "state": "AWAKE",
                "uptime_seconds": uptime,
                "requests_received": PING_STATS["requests_handled"],
                "timestamp": now_str
            }
            self.wfile.write(json.dumps(response, indent=2).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

def run_keepalive_server(port=8502):
    try:
        server = HTTPServer(("0.0.0.0", port), KeepAliveHandler)
        server.serve_forever()
    except Exception as e:
        # Port might be taken or non-critical
        pass

def run_self_ping_worker(target_url=None, interval=300):
    """Periodically sends HTTP GET to target_url to keep cloud containers awake."""
    import urllib.request
    while True:
        time.sleep(interval)
        url_to_ping = target_url or os.environ.get("STREAMLIT_APP_URL") or "http://127.0.0.1:8502/ping"
        try:
            req = urllib.request.Request(url_to_ping, headers={"User-Agent": "SentinelKeepAlive/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                status_code = resp.getcode()
            with STATS_LOCK:
                PING_STATS["self_pings_sent"] += 1
                now_str = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
                PING_STATS["last_self_ping"] = now_str
                PING_STATS["self_ping_status"] = f"OK ({status_code}) at {now_str}"
        except Exception as err:
            with STATS_LOCK:
                PING_STATS["self_ping_status"] = f"Failed ({type(err).__name__})"

# Start background server once globally
@st.cache_resource
def initialize_background_daemons():
    ping_port = int(os.environ.get("PING_PORT", 8502))
    server_thread = threading.Thread(target=run_keepalive_server, args=(ping_port,), daemon=True)
    server_thread.start()

    self_ping_thread = threading.Thread(target=run_self_ping_worker, args=(None, 240), daemon=True)
    self_ping_thread.start()
    return {"server_thread": server_thread, "ping_port": ping_port}

DAEMON_INFO = initialize_background_daemons()

# -----------------------------------------------------------------------------
# 3. TAILWIND CSS & ORGANIC OFF-WHITE DESIGN SYSTEM
# -----------------------------------------------------------------------------
st.markdown("""
<!-- Load Tailwind CSS via CDN -->
<script src="https://cdn.tailwindcss.com"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700&display=swap" rel="stylesheet">

<style>
/* Global Off-White Typography & Palette */
:root {
    --bg-canvas: #FBFBF9;
    --card-surface: #FFFFFF;
    --card-subtle: #F6F5F0;
    --border-warm: #E7E5DF;
    --border-light: #ECEAE4;
    --text-primary: #1C1917;
    --text-secondary: #57534E;
    --text-muted: #78716C;
    --accent-sage: #2E6F40;
    --accent-sage-bg: #EEF7F1;
    --accent-sage-border: #9DC3A7;
    --accent-terracotta: #991B1B;
    --accent-terracotta-bg: #FEF2F2;
    --accent-terracotta-border: #F87171;
}

html, body, [class*="st-"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text-primary);
}

.stApp {
    background-color: var(--bg-canvas);
}

/* Hide Streamlit default chrome clutter */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

/* Custom Card Classes */
.sentinel-card {
    background-color: #FFFFFF;
    border: 1px solid #E8E6DF;
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 2px 10px rgba(28, 25, 23, 0.03), 0 1px 3px rgba(28, 25, 23, 0.02);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.sentinel-card:hover {
    border-color: #D6D3C9;
    box-shadow: 0 6px 18px rgba(28, 25, 23, 0.05);
}

.sentinel-card-warm {
    background: linear-gradient(145deg, #FAF8F5 0%, #F3EFE9 100%);
    border: 1px solid #E6E2D8;
    border-radius: 16px;
    padding: 1.5rem;
}

/* Animations */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes pulseWarm {
    0% { transform: scale(1); opacity: 0.9; }
    50% { transform: scale(1.04); opacity: 1; }
    100% { transform: scale(1); opacity: 0.9; }
}

@keyframes scanProgress {
    0% { width: 0%; }
    100% { width: 100%; }
}

.fade-in {
    animation: fadeIn 0.4s ease-out forwards;
}

.pulse-dot {
    animation: pulseWarm 2s infinite ease-in-out;
}

/* Custom Progress Meter */
.risk-meter-track {
    background: #EAE8E1;
    border-radius: 9999px;
    height: 12px;
    overflow: hidden;
    position: relative;
}
.risk-meter-fill {
    height: 100%;
    border-radius: 9999px;
    transition: width 0.8s cubic-bezier(0.16, 1, 0.3, 1);
}

/* Streamlit Button & Text Area Customization */
div.stButton > button:first-child {
    background: #1C1917;
    color: #FBFBF9;
    font-weight: 600;
    border-radius: 10px;
    border: none;
    padding: 0.65rem 1.4rem;
    transition: all 0.2s ease;
    box-shadow: 0 2px 6px rgba(28, 25, 23, 0.15);
}
div.stButton > button:first-child:hover {
    background: #2E2A27;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(28, 25, 23, 0.2);
    color: #FFFFFF;
}

div.stTextArea textarea {
    border-radius: 12px;
    border: 1px solid #D6D3C9;
    background-color: #FFFFFF;
    color: #1C1917;
    font-size: 0.95rem;
    line-height: 1.5;
    padding: 0.85rem;
    transition: border-color 0.2s;
}
div.stTextArea textarea:focus {
    border-color: #78716C;
    box-shadow: 0 0 0 2px rgba(120, 113, 108, 0.15);
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. MODEL LOADING & TEXT PREPROCESSING
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

@st.cache_resource(show_spinner="Loading deep learning model and tokenizer...")
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
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    LOAD_ERROR = str(e)

def preprocess_email_text(raw_text):
    """Exact pipeline matching training notebook: lower -> remove punct -> remove stopwords -> remove URLs."""
    # Lowercase
    text = raw_text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    # Tokenize words & remove stopwords
    tokens = text.split()
    filtered_tokens = [w for w in tokens if w not in DEFAULT_STOPWORDS]
    cleaned_text = " ".join(filtered_tokens)
    return cleaned_text, filtered_tokens

def analyze_email(raw_text):
    """Performs inference and extracts diagnostic signals."""
    cleaned_text, tokens = preprocess_email_text(raw_text)
    seq = TOKENIZER.texts_to_sequences([cleaned_text])
    padded = pad_sequences(seq, maxlen=CONFIG["max_length"], padding="post")
    
    # Run GRU prediction
    prob = float(MODEL.predict(padded, verbose=0)[0][0])
    is_spam = prob >= 0.5
    verdict = "Spam" if is_spam else "Ham"
    confidence = (prob if is_spam else (1.0 - prob)) * 100
    
    # Identify high risk trigger words
    detected_spam_words = [w for w in tokens if w in HIGH_RISK_SPAM_WORDS]
    
    return {
        "raw_text": raw_text,
        "cleaned_text": cleaned_text,
        "token_count": len(tokens),
        "raw_char_count": len(raw_text),
        "raw_word_count": len(raw_text.split()),
        "probability": prob,
        "verdict": verdict,
        "is_spam": is_spam,
        "confidence": confidence,
        "trigger_words": list(set(detected_spam_words)),
        "padded_sequence_length": CONFIG["max_length"],
        "tokens_recognized": len(seq[0]) if seq else 0
    }

# -----------------------------------------------------------------------------
# 5. SIDEBAR: ANTI-SLEEP CONTROL CENTER & QUICK STATUS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="mb-4">
        <div class="flex items-center space-x-2">
            <span class="w-3 h-3 rounded-full bg-emerald-600 pulse-dot inline-block"></span>
            <span class="text-xs font-bold tracking-wider text-stone-500 uppercase">System Status</span>
        </div>
        <h2 class="text-xl font-bold text-stone-900 tracking-tight mt-1">Sentinel Core</h2>
        <p class="text-xs text-stone-500">GRU Deep Learning • v2.4 Release</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Live Uptime & Keep-Alive Box
    uptime_seconds = int(time.time() - APP_START_TIME)
    uptime_min = uptime_seconds // 60
    uptime_hrs = uptime_min // 60
    uptime_display = f"{uptime_hrs}h {uptime_min % 60}m {uptime_seconds % 60}s"
    
    st.markdown(f"""
    <div class="bg-stone-100 rounded-xl p-3 border border-stone-200 text-xs text-stone-700 mb-4">
        <div class="flex justify-between py-1 border-b border-stone-200">
            <span class="text-stone-500">Cloud Health Route</span>
            <span class="font-semibold text-emerald-800">Port {DAEMON_INFO['ping_port']} (/ping)</span>
        </div>
        <div class="flex justify-between py-1 border-b border-stone-200">
            <span class="text-stone-500">Process Uptime</span>
            <span class="font-mono font-medium">{uptime_display}</span>
        </div>
        <div class="flex justify-between py-1 border-b border-stone-200">
            <span class="text-stone-500">Keep-Alive Status</span>
            <span class="text-emerald-700 font-semibold">Hot & Active</span>
        </div>
        <div class="flex justify-between py-1">
            <span class="text-stone-500">Pings Received</span>
            <span class="font-semibold">{PING_STATS['requests_handled']} requests</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Navigation
    st.markdown("<p class='text-xs font-semibold text-stone-400 uppercase tracking-wider mb-2'>Workspaces</p>", unsafe_allow_html=True)
    selected_tab = st.radio(
        label="Select Workspace",
        options=[
            "🔍 Email Threat Inspector",
            "📂 Batch Classifier",
            "⚡ Keep-Alive & Anti-Sleep",
            "🧠 Architecture & Benchmarks"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("<hr class='my-4 border-stone-200'>", unsafe_allow_html=True)
    
    # Model Specs Card
    st.markdown("""
    <div class="text-xs text-stone-600 space-y-1">
        <div class="font-semibold text-stone-800 mb-1">Architecture Overview</div>
        <p>• <b>Model:</b> Recurrent GRU (64 Units)</p>
        <p>• <b>Embedding Dim:</b> 32</p>
        <p>• <b>Max Vocabulary:</b> 7,000 Words</p>
        <p>• <b>Sequence Pad:</b> 500 Tokens</p>
        <p>• <b>F1-Score:</b> 98.72% (Top Benchmark)</p>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 6. HEADER BANNER (HUMAN-CRAFTED OFF-WHITE LUXURY DESIGN)
# -----------------------------------------------------------------------------
st.markdown("""
<div class="sentinel-card-warm mb-6 fade-in">
    <div class="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
            <div class="inline-flex items-center space-x-2 bg-stone-200/80 px-2.5 py-1 rounded-full text-xs font-medium text-stone-700 mb-2">
                <span>🛡️</span>
                <span>Neural Threat Analysis</span>
                <span class="text-stone-400">•</span>
                <span class="text-emerald-800 font-semibold">GRU Engine Active</span>
            </div>
            <h1 class="text-2xl md:text-3xl font-extrabold text-stone-900 tracking-tight">
                Sentinel Mail Intelligence
            </h1>
            <p class="text-sm text-stone-600 mt-1 max-w-2xl">
                Advanced recurrent deep learning system trained on 83,000+ verified enterprise communications to classify spam, fraud, and phishing vectors with 98.6% precision.
            </p>
        </div>
        <div class="flex items-center gap-3">
            <div class="text-right hidden sm:block">
                <div class="text-xs text-stone-500 font-medium">Keep-Alive Engine</div>
                <div class="text-xs font-bold text-emerald-800 bg-emerald-100/70 border border-emerald-200 px-2 py-0.5 rounded-md inline-block">
                    ● Anti-Sleep Daemon Active
                </div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Check model load status
if not MODEL_LOADED:
    st.error(f"⚠️ Model artifacts could not be loaded: {LOAD_ERROR}")
    st.stop()

# -----------------------------------------------------------------------------
# TAB 1: EMAIL THREAT INSPECTOR
# -----------------------------------------------------------------------------
if selected_tab == "🔍 Email Threat Inspector":
    col_input, col_meta = st.columns([7, 3])
    
    with col_input:
        st.markdown("<h3 class='text-base font-bold text-stone-800 mb-1'>Analyze Email Content</h3>", unsafe_allow_html=True)
        st.markdown("<p class='text-xs text-stone-500 mb-3'>Select a preset or paste any email subject/body for instant GRU inference.</p>", unsafe_allow_html=True)
        
        # Presets row
        preset_choice = st.selectbox(
            "Load Sample Template",
            [
                "-- Choose a sample email preset --",
                "🚨 Urgent Phishing: Account Suspended (Spam)",
                "💰 Foreign Lottery & Mega Cash Award (Spam)",
                "💊 Prescription Pharma Discount (Spam)",
                "💼 Team Quarterly Planning & Sync (Ham)",
                "📦 Courier Package Delivery Notification (Ham)"
            ],
            index=0
        )
        
        sample_texts = {
            "🚨 Urgent Phishing: Account Suspended (Spam)": 
                "URGENT SECURITY ALERT: Your corporate email account has been flagged for multiple unauthorized login attempts. To prevent immediate permanent suspension, you must verify your identity within 24 hours. Click here to confirm your password and restore full security privileges: http://secure-auth-login-portal.net/verify",
            "💰 Foreign Lottery & Mega Cash Award (Spam)":
                "Congratulations Dear Winner! You have won the official sum of $2,500,000 USD in our annual international lottery sweepstakes. To facilitate the immediate wire transfer of your award, send your full bank account, passport number, and phone number to our claims agent right now. Claim prize today!",
            "💊 Prescription Pharma Discount (Spam)":
                "Get your discount medications online with no prior prescription required! High quality viagra, cialis, valium, and pain relief pills at lowest guaranteed prices with free discreet overnight shipping. Order yours now at our official pharmacy portal.",
            "💼 Team Quarterly Planning & Sync (Ham)":
                "Hi Alex and Team,\n\nFollowing up on our sprint review, please find attached the draft roadmap for Q3 deliverables. Let's meet this Thursday at 2:30 PM in Conference Room B to finalize team allocations and KPIs. Let me know if that time works for you.\n\nBest regards,\nSarah",
            "📦 Courier Package Delivery Notification (Ham)":
                "Hello Customer,\n\nYour recent shipment with tracking ID #940010023456 has been dispatched from our regional distribution hub and is scheduled for contactless delivery tomorrow between 1:00 PM and 4:00 PM. No signature required. Thank you for your business."
        }
        
        initial_val = sample_texts.get(preset_choice, "")
        
        email_content = st.text_area(
            "Email Body",
            value=initial_val,
            height=200,
            placeholder="Paste raw email text, headers, or messages here...",
            label_visibility="collapsed"
        )
        
        btn_col1, btn_col2 = st.columns([3, 7])
        with btn_col1:
            analyze_button = st.button("⚡ Run Threat Detection", use_container_width=True)
        with btn_col2:
            st.markdown("<p class='text-xs text-stone-500 mt-2'>Processed locally with GRU recurrent units • Zero data shared externally</p>", unsafe_allow_html=True)
    
    with col_meta:
        st.markdown("<h3 class='text-base font-bold text-stone-800 mb-1'>Live Buffer Metrics</h3>", unsafe_allow_html=True)
        word_count = len(email_content.split()) if email_content.strip() else 0
        char_count = len(email_content)
        url_matches = len(re.findall(r"http\S+|www\S+", email_content))
        
        st.markdown(f"""
        <div class="sentinel-card space-y-3 text-xs mb-3">
            <div class="flex justify-between items-center pb-2 border-b border-stone-100">
                <span class="text-stone-500">Character Length</span>
                <span class="font-bold text-stone-800">{char_count} chars</span>
            </div>
            <div class="flex justify-between items-center pb-2 border-b border-stone-100">
                <span class="text-stone-500">Word Count</span>
                <span class="font-bold text-stone-800">{word_count} words</span>
            </div>
            <div class="flex justify-between items-center pb-2 border-b border-stone-100">
                <span class="text-stone-500">Embedded Hyperlinks</span>
                <span class="font-bold {'text-rose-700' if url_matches > 0 else 'text-stone-800'}">{url_matches} found</span>
            </div>
            <div class="flex justify-between items-center">
                <span class="text-stone-500">Max Sequence Buffer</span>
                <span class="font-bold text-stone-800">500 tokens</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="bg-amber-50/70 border border-amber-200/80 rounded-xl p-3 text-xs text-amber-900 leading-relaxed">
            <b>Detection Advisory:</b> The GRU model analyzes structural semantics, token frequencies, and deceptive linguistic cues across sequences.
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # ANALYSIS RESULTS SECTION
    # -------------------------------------------------------------------------
    if email_content.strip():
        if analyze_button or preset_choice != "-- Choose a sample email preset --":
            res = analyze_email(email_content)
            
            st.markdown("<div class='my-6 border-t border-stone-200'></div>", unsafe_allow_html=True)
            
            # Big Verdict Banner
            if res["is_spam"]:
                verdict_card = f"""
                <div class="rounded-2xl p-6 border border-rose-300 bg-gradient-to-r from-rose-50/90 via-red-50/60 to-rose-50/90 text-rose-950 shadow-sm fade-in mb-6">
                    <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                        <div class="flex items-center space-x-3">
                            <span class="text-3xl">🚨</span>
                            <div>
                                <div class="text-xs uppercase tracking-widest font-bold text-rose-700">Classification Verdict</div>
                                <h2 class="text-2xl font-extrabold text-rose-900 tracking-tight">HIGH-RISK SPAM DETECTED</h2>
                            </div>
                        </div>
                        <div class="bg-rose-100/80 border border-rose-200 px-4 py-2 rounded-xl text-right">
                            <div class="text-xs text-rose-700 font-semibold">Spam Probability</div>
                            <div class="text-2xl font-extrabold text-rose-900">{res['confidence']:.2f}%</div>
                        </div>
                    </div>
                    <p class="text-xs text-rose-800/80 mt-3">
                        The neural sequence processor detected recurring deception markers and high spam probability. Treat links and attachments as untrusted.
                    </p>
                </div>
                """
            else:
                verdict_card = f"""
                <div class="rounded-2xl p-6 border border-emerald-300 bg-gradient-to-r from-emerald-50/90 via-teal-50/60 to-emerald-50/90 text-emerald-950 shadow-sm fade-in mb-6">
                    <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                        <div class="flex items-center space-x-3">
                            <span class="text-3xl">🛡️</span>
                            <div>
                                <div class="text-xs uppercase tracking-widest font-bold text-emerald-700">Classification Verdict</div>
                                <h2 class="text-2xl font-extrabold text-emerald-900 tracking-tight">LEGITIMATE COMMUNICATION (HAM)</h2>
                            </div>
                        </div>
                        <div class="bg-emerald-100/80 border border-emerald-200 px-4 py-2 rounded-xl text-right">
                            <div class="text-xs text-emerald-700 font-semibold">Clean Confidence</div>
                            <div class="text-2xl font-extrabold text-emerald-900">{res['confidence']:.2f}%</div>
                        </div>
                    </div>
                    <p class="text-xs text-emerald-800/80 mt-3">
                        Content exhibits natural phrasing patterns consistent with genuine enterprise or personal correspondence.
                    </p>
                </div>
                """
            st.markdown(verdict_card, unsafe_allow_html=True)
            
            # Risk Gauge & Diagnostics
            gauge_col1, gauge_col2 = st.columns([6, 4])
            
            with gauge_col1:
                st.markdown("<h4 class='text-sm font-bold text-stone-800 mb-2'>Spam Probability Spectrum</h4>", unsafe_allow_html=True)
                prob_pct = res["probability"] * 100
                bar_color = "linear-gradient(90deg, #34D399 0%, #FBBF24 50%, #EF4444 100%)"
                
                st.markdown(f"""
                <div class="sentinel-card">
                    <div class="flex justify-between text-xs font-semibold text-stone-600 mb-1.5">
                        <span>Safe (0%)</span>
                        <span class="text-stone-900 font-bold">{prob_pct:.1f}% Risk Score</span>
                        <span>Threat (100%)</span>
                    </div>
                    <div class="risk-meter-track">
                        <div class="risk-meter-fill" style="width: {prob_pct}%; background: {bar_color};"></div>
                    </div>
                    <div class="mt-4 grid grid-cols-3 gap-2 text-center text-xs">
                        <div class="bg-stone-50 p-2 rounded-lg border border-stone-100">
                            <div class="text-stone-400">Model Output</div>
                            <div class="font-mono font-bold text-stone-800">{res['probability']:.4f}</div>
                        </div>
                        <div class="bg-stone-50 p-2 rounded-lg border border-stone-100">
                            <div class="text-stone-400">Tokens Ingested</div>
                            <div class="font-mono font-bold text-stone-800">{res['tokens_recognized']}</div>
                        </div>
                        <div class="bg-stone-50 p-2 rounded-lg border border-stone-100">
                            <div class="text-stone-400">Stopwords Removed</div>
                            <div class="font-mono font-bold text-stone-800">{res['raw_word_count'] - res['token_count']}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with gauge_col2:
                st.markdown("<h4 class='text-sm font-bold text-stone-800 mb-2'>Trigger Keywords Detected</h4>", unsafe_allow_html=True)
                if res["trigger_words"]:
                    tags_html = "".join([
                        f"<span class='inline-block bg-rose-100 text-rose-800 border border-rose-200 px-2.5 py-1 rounded-md text-xs font-medium mr-1.5 mb-1.5'>⚠ {word}</span>"
                        for word in res["trigger_words"]
                    ])
                    st.markdown(f"""
                    <div class="sentinel-card">
                        <div class="text-xs text-stone-500 mb-2">High-frequency terms matching spam corpora:</div>
                        <div class="flex flex-wrap">{tags_html}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="sentinel-card text-center py-6">
                        <span class="text-2xl">🌱</span>
                        <div class="text-xs font-semibold text-emerald-800 mt-1">No Blatant Trigger Keywords Found</div>
                        <p class="text-xs text-stone-400 mt-0.5">Linguistic structure verified organically by GRU sequence memory.</p>
                    </div>
                    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 2: BATCH EMAIL CLASSIFIER
# -----------------------------------------------------------------------------
elif selected_tab == "📂 Batch Classifier":
    st.markdown("<h3 class='text-lg font-bold text-stone-900 mb-1'>Batch Email Screening</h3>", unsafe_allow_html=True)
    st.markdown("<p class='text-xs text-stone-500 mb-4'>Upload a CSV file containing an email column, or paste multiple messages separated by line breaks.</p>", unsafe_allow_html=True)
    
    upload_file = st.file_uploader("Upload CSV file with 'text' column", type=["csv"])
    
    if upload_file is not None:
        try:
            batch_df = pd.read_csv(upload_file)
            st.success(f"Loaded {len(batch_df)} records successfully.")
            
            target_col = None
            for col in ["text", "email", "body", "content", "message"]:
                if col in batch_df.columns:
                    target_col = col
                    break
            
            if target_col is None:
                target_col = st.selectbox("Select the column containing email text", batch_df.columns)
            
            if st.button("🚀 Classify Entire Batch"):
                progress_bar = st.progress(0)
                predictions = []
                probabilities = []
                
                total = len(batch_df)
                for idx, row in batch_df.iterrows():
                    text_val = str(row[target_col])
                    res = analyze_email(text_val)
                    predictions.append(res["verdict"])
                    probabilities.append(round(res["probability"], 4))
                    if (idx + 1) % max(1, total // 20) == 0:
                        progress_bar.progress((idx + 1) / total)
                
                progress_bar.progress(1.0)
                batch_df["Prediction"] = predictions
                batch_df["Spam_Probability"] = probabilities
                
                # Metrics overview
                spam_count = (batch_df["Prediction"] == "Spam").sum()
                ham_count = (batch_df["Prediction"] == "Ham").sum()
                
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("Total Scanned", total)
                col_m2.metric("Spam Detected", f"{spam_count} ({spam_count/total*100:.1f}%)")
                col_m3.metric("Clean (Ham)", f"{ham_count} ({ham_count/total*100:.1f}%)")
                
                st.dataframe(batch_df.head(50), use_container_width=True)
                
                # CSV Download
                csv_data = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Classified CSV",
                    data=csv_data,
                    file_name="sentinel_classified_results.csv",
                    mime="text/csv"
                )
        except Exception as ex:
            st.error(f"Error processing CSV: {ex}")
    else:
        st.markdown("""
        <div class="sentinel-card text-center py-10 my-4">
            <div class="text-4xl mb-2">📁</div>
            <div class="font-bold text-stone-800 text-sm">No CSV Uploaded Yet</div>
            <p class="text-xs text-stone-500 max-w-sm mx-auto mt-1">Upload a CSV file containing an email body column (e.g. 'text', 'body') to perform bulk classification.</p>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 3: KEEP-ALIVE & CLOUD SLEEP PREVENTION HUB
# -----------------------------------------------------------------------------
elif selected_tab == "⚡ Keep-Alive & Anti-Sleep":
    st.markdown("<h3 class='text-lg font-bold text-stone-900 mb-1'>Cloud Sleep Prevention & Keep-Alive Hub</h3>", unsafe_allow_html=True)
    st.markdown("<p class='text-xs text-stone-500 mb-4'>Prevent inactivity shutdowns on Streamlit Community Cloud, Render, or Hugging Face Spaces.</p>", unsafe_allow_html=True)
    
    uptime_sec = int(time.time() - APP_START_TIME)
    
    col_k1, col_k2, col_k3 = st.columns(3)
    with col_k1:
        st.markdown(f"""
        <div class="sentinel-card">
            <div class="text-xs text-stone-500">Daemon Server Port</div>
            <div class="text-xl font-bold text-stone-900 mt-1">:{DAEMON_INFO['ping_port']}</div>
            <div class="text-xs text-emerald-700 font-medium mt-1">● Actively Listening</div>
        </div>
        """, unsafe_allow_html=True)
    with col_k2:
        st.markdown(f"""
        <div class="sentinel-card">
            <div class="text-xs text-stone-500">Inbound Pings Handled</div>
            <div class="text-xl font-bold text-stone-900 mt-1">{PING_STATS['requests_handled']}</div>
            <div class="text-xs text-stone-400 mt-1">Last: {PING_STATS['last_ping'] or 'Awaiting initial ping'}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_k3:
        st.markdown(f"""
        <div class="sentinel-card">
            <div class="text-xs text-stone-500">Continuous Uptime</div>
            <div class="text-xl font-bold text-stone-900 mt-1">{uptime_sec // 60}m {uptime_sec % 60}s</div>
            <div class="text-xs text-emerald-700 font-medium mt-1">Process Stays Hot</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<div class='my-4'></div>", unsafe_allow_html=True)
    
    # How It Works & Configuration Guide
    st.markdown("""
    <div class="sentinel-card-warm mb-6">
        <h4 class="font-bold text-stone-900 text-sm mb-2">Why do Streamlit apps sleep?</h4>
        <p class="text-xs text-stone-600 leading-relaxed mb-3">
            Streamlit Community Cloud and free hosting platforms put applications to sleep if no HTTP traffic arrives for several minutes. When asleep, user requests take 30-60 seconds to cold-boot. 
            <b>Sentinel solves this by embedding two layers of protection:</b>
        </p>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
            <div class="bg-white/80 p-3 rounded-xl border border-stone-200">
                <div class="font-bold text-stone-800 mb-1">Layer 1: Internal Health & Ping Route</div>
                <p class="text-stone-600">A lightweight background HTTP thread responds to <code>/ping</code> and <code>/health</code> endpoints with instant 200 OK telemetry, allowing any uptime pinger to keep it awake.</p>
            </div>
            <div class="bg-white/80 p-3 rounded-xl border border-stone-200">
                <div class="font-bold text-stone-800 mb-1">Layer 2: 24/7 External Ping Integration</div>
                <p class="text-stone-600">Connect free monitoring services (such as <b>UptimeRobot</b> or <b>Cron-Job.org</b>) to ping your public URL every 5 minutes completely for free.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h4 class='text-sm font-bold text-stone-800 mb-2'>Configure Free 5-Minute Keep-Alive Monitor</h4>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="sentinel-card space-y-3 text-xs">
        <div class="font-semibold text-stone-800">3-Step Free Setup (Guarantees 100% Awake Cloud App):</div>
        <ol class="list-decimal list-inside space-y-2 text-stone-600">
            <li>Register a free account on <a href="https://uptimerobot.com" target="_blank" class="text-stone-900 underline font-semibold">UptimeRobot.com</a> or <a href="https://cron-job.org" target="_blank" class="text-stone-900 underline font-semibold">Cron-Job.org</a>.</li>
            <li>Click <b>Add New Monitor</b> &rarr; Choose <b>HTTP(s)</b>.</li>
            <li>Set the URL to your deployed Streamlit app URL with the native health path:
                <br><code class="bg-stone-100 px-2 py-1 rounded text-stone-800 font-mono inline-block my-1">https://&lt;your-app-name&gt;.streamlit.app/_stcore/health</code>
            </li>
            <li>Set monitoring interval to <b>Every 5 minutes</b> &rarr; Save. Your app will now run 24/7 without ever sleeping!</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 4: MODEL ARCHITECTURE & BENCHMARKS
# -----------------------------------------------------------------------------
elif selected_tab == "🧠 Architecture & Benchmarks":
    st.markdown("<h3 class='text-lg font-bold text-stone-900 mb-1'>Model Evaluation & Benchmarks</h3>", unsafe_allow_html=True)
    st.markdown("<p class='text-xs text-stone-500 mb-4'>Rigorous evaluation comparison across SimpleRNN, LSTM, and the deployed GRU model trained on 83,448 emails.</p>", unsafe_allow_html=True)
    
    col_b1, col_b2, col_b3 = st.columns(3)
    
    with col_b1:
        st.markdown("""
        <div class="sentinel-card">
            <div class="text-xs font-semibold text-stone-400 uppercase">Baseline Model</div>
            <div class="text-xl font-bold text-stone-800 mt-1">Simple RNN</div>
            <div class="mt-3 space-y-1 text-xs">
                <div class="flex justify-between"><span>Accuracy:</span><span class="font-mono font-bold">97.69%</span></div>
                <div class="flex justify-between"><span>Spam Precision:</span><span class="font-mono">97.82%</span></div>
                <div class="flex justify-between"><span>Spam Recall:</span><span class="font-mono">97.80%</span></div>
                <div class="flex justify-between"><span>Spam F1:</span><span class="font-mono">97.81%</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_b2:
        st.markdown("""
        <div class="sentinel-card">
            <div class="text-xs font-semibold text-stone-400 uppercase">Long-Term Model</div>
            <div class="text-xl font-bold text-stone-800 mt-1">LSTM</div>
            <div class="mt-3 space-y-1 text-xs">
                <div class="flex justify-between"><span>Accuracy:</span><span class="font-mono font-bold">98.48%</span></div>
                <div class="flex justify-between"><span>Spam Precision:</span><span class="font-mono">98.38%</span></div>
                <div class="flex justify-between"><span>Spam Recall:</span><span class="font-mono">98.74%</span></div>
                <div class="flex justify-between"><span>Spam F1:</span><span class="font-mono">98.56%</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_b3:
        st.markdown("""
        <div class="sentinel-card border-2 border-stone-800 bg-stone-900 text-white shadow-md">
            <div class="flex justify-between items-center">
                <span class="text-xs font-semibold text-stone-300 uppercase">Deployed Model</span>
                <span class="bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 text-[10px] px-2 py-0.5 rounded-full font-bold">BEST</span>
            </div>
            <div class="text-xl font-bold text-white mt-1">Gated Recurrent (GRU)</div>
            <div class="mt-3 space-y-1 text-xs text-stone-200">
                <div class="flex justify-between"><span>Accuracy:</span><span class="font-mono font-bold text-emerald-400">98.65%</span></div>
                <div class="flex justify-between"><span>Spam Precision:</span><span class="font-mono text-emerald-400">98.75%</span></div>
                <div class="flex justify-between"><span>Spam Recall:</span><span class="font-mono">98.69%</span></div>
                <div class="flex justify-between"><span>Spam F1:</span><span class="font-mono text-emerald-400 font-bold">98.72%</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<div class='my-6'></div>", unsafe_allow_html=True)
    
    # Detailed Dataset & Pipeline Specs
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown("""
        <div class="sentinel-card">
            <h4 class="font-bold text-stone-900 text-sm mb-3">Dataset & Preprocessing Pipeline</h4>
            <div class="space-y-2 text-xs text-stone-600">
                <p>• <b>Total Training Corpus:</b> 83,448 labeled emails (Spam = 1, Ham = 0).</p>
                <p>• <b>Cleaning:</b> Lowercase normalization, complete punctuation removal via ASCII translation, URL link neutralization.</p>
                <p>• <b>Stopwords:</b> Comprehensive English stopword filter isolating content-bearing lexical terms.</p>
                <p>• <b>Vocabulary Size:</b> Top 7,000 most frequent tokens fitted on training partition.</p>
                <p>• <b>Sequence Length:</b> 500 tokens with post-padding zero masking.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_d2:
        st.markdown("""
        <div class="sentinel-card">
            <h4 class="font-bold text-stone-900 text-sm mb-3">Layer Topology & Parameters</h4>
            <div class="space-y-2 text-xs text-stone-600 font-mono">
                <div class="bg-stone-50 p-2 rounded border border-stone-200">
                    <span class="text-stone-900 font-bold">Layer 1: Embedding</span> (input_dim=7000, output_dim=32, mask_zero=True)
                </div>
                <div class="bg-stone-50 p-2 rounded border border-stone-200">
                    <span class="text-stone-900 font-bold">Layer 2: GRU</span> (units=64, return_sequences=False)
                </div>
                <div class="bg-stone-50 p-2 rounded border border-stone-200">
                    <span class="text-stone-900 font-bold">Layer 3: Dense</span> (units=1, activation='sigmoid')
                </div>
                <div class="text-stone-500 font-sans text-xs pt-1">
                    Optimizer: Adam • Loss: Binary Crossentropy • Metric: Accuracy
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
