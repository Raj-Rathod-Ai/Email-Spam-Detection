# 🛡️ Sentinel • Email Spam & Threat Detection System

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.0+-red.svg)](https://keras.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Deep Learning Natural Language Processing (NLP) system designed to detect and classify unsolicited, phishing, and fraudulent emails with **98.65% test accuracy**. The project benchmarks **Simple RNN**, **LSTM**, and **GRU (Gated Recurrent Unit)** architectures on a large-scale corpus of **83,448 emails**, selecting the GRU model as the premier production engine.

Includes a web interface built with **Streamlit** and **Tailwind CSS**, featuring:
- A human-crafted **off-white, sky-blue, and light-green** aesthetic
- An animated **5–8 second neural reading / analysis scanner**
- A **silent background keep-alive daemon** preventing cloud inactivity sleep on Streamlit Community Cloud, Render, or Hugging Face.

---

## 📊 Model Comparison & Benchmarks

All three recurrent neural architectures were trained on identical partitions (67% Training / 33% Validation) across 10 epochs using Adam optimization and Binary Crossentropy loss:

| Model Architecture | Test Accuracy | Spam Precision | Spam Recall | Spam F1-Score | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Simple RNN** | 97.69% | 97.82% | 97.80% | 0.9781 | Baseline |
| **LSTM** (Long Short-Term Memory) | 98.48% | 98.38% | 98.74% | 0.9856 | High Memory |
| **GRU** (Gated Recurrent Unit) | **98.65%** | **98.75%** | **98.69%** | **0.9872** | **Best / Deployed** 🏆 |

### Why GRU Outperformed:
- **Faster Convergence:** With fewer parameters than LSTM (combining cell state and hidden state into a single reset/update gate mechanism), the GRU model trained faster with less risk of overfitting.
- **Superior Long-Sequence Modeling:** GRU effectively avoided vanishing gradients across the 500-token sequence window, achieving the highest precision (**98.75%**) in identifying malicious vectors.

---

## 🧠 Neural Architecture Pipeline

```
Raw Email Text
      │
      ▼
Text Preprocessing
  ├─ Lowercase normalization
  ├─ ASCII punctuation stripping
  ├─ URL neutralization (Regex)
  └─ Stopword filtering (isolates content-bearing tokens)
      │
      ▼
Keras Tokenizer (Vocabulary: 7,000 top words)
      │
      ▼
Sequence Post-Padding (maxlen = 500 tokens)
      │
      ▼
Embedding Layer (input_dim=7000, output_dim=32, mask_zero=True)
      │
      ▼
GRU Layer (units=64, return_sequences=False)
      │
      ▼
Dense Output Layer (units=1, activation='sigmoid')
      │
      ▼
Threat Probability (0.0 to 1.0) ──> [ Ham (<0.5) | Spam (>=0.5) ]
```

---

## ✨ Web Application Highlights

- **Aesthetic Palette:** Crafted with an off-white canvas (`#F8FAFC`), crisp sky-blue action accents (`#0284C7`), and calming light-green indicators (`#16A34A` / `#DCFCE7`) for authentic enterprise correspondence.
- **Reading & Analysis Animation:** When an email is submitted, the UI runs a realistic 5–8 second multi-phase scanning animation simulating token parsing, stopword extraction, recurrent gate passage, and probability calculation.
- **Real-Time Diagnostics:** Displays extracted spam keywords (e.g., `account`, `unauthorized`, `urgent`, `verify`, `viagra`, `lottery`), character/word lengths, embedded URL counts, and risk gauges.
- **Batch CSV Screening:** Upload a CSV file of emails to screen hundreds of records at once and export results directly.
- **Silent Cloud Sleep Prevention:** Features an embedded HTTP health endpoint (`/ping` & `/health` on port 8502) and self-ping daemon that prevents cold-boot hibernation on cloud hosting platforms.

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### 2. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/Raj-Rathod-Ai/Email-Spam-Detection.git
cd Email-Spam-Detection
pip install -r requirements.txt
```

### 3. Launching the Web App
Run the Streamlit application:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🌐 Cloud Deployment (Streamlit Community Cloud / Render)

### Deploying to Streamlit Cloud:
1. Push this repository to your GitHub account.
2. Sign in to [share.streamlit.io](https://share.streamlit.io).
3. Connect your GitHub repository:
   - **Repository:** `Raj-Rathod-Ai/Email-Spam-Detection`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **Deploy!**

### Ensuring 100% 24/7 Uptime (No Cloud Sleep):
1. Sign up for a free monitor on [UptimeRobot](https://uptimerobot.com) or [Cron-Job.org](https://cron-job.org).
2. Create an **HTTP(s)** monitor pointing to your app's health route:
   ```
   https://<your-app-name>.streamlit.app/_stcore/health
   ```
3. Set the interval to **Every 5 minutes**. Your app will remain hot and responsive around the clock with zero cold boots.

---

## 📁 Repository Structure

```
├── .streamlit/
│   └── config.toml           # Streamlit theme (off-white, sky-blue) & server config
├── app.py                    # Streamlit web application with neural scanner & keep-alive
├── config.pkl                # Hyperparameters (vocab: 7000, maxlen: 500, embed: 32)
├── gru_model.keras           # Trained GRU model weights and architecture
├── label_mapping.pkl         # Target labels mapping {0: 'Ham', 1: 'Spam'}
├── tokenizer.pkl             # Fitted Keras text tokenizer
├── Email_Spam_detection.ipynb# Jupyter Notebook containing dataset exploration & model training
├── requirements.txt          # Production dependencies
├── .gitignore                # Excludes large CSV datasets and cache files
└── README.md                 # Project documentation
```

---

## 👨‍💻 Author
- **Raj Rathod** ([@Raj-Rathod-Ai](https://github.com/Raj-Rathod-Ai))

Feel free to star ⭐ the repository if you found this helpful!
