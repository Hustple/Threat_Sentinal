import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time

# Configuration
MAX_WORDS = 10000
MAX_LEN = 200

st.set_page_config(page_title="URL Threat Analyzer", layout="centered")
st.title("🔍 URL Threat Analyzer")
st.markdown("Paste a full URL below to check if it's **safe** or potentially **malicious**")

# Sidebar Instructions
with st.sidebar:
    st.header("Instructions")
    st.write("1. Paste a full URL in the textbox.")
    st.write("2. Or click an example URL to auto-fill and analyze.")

# File Check
if not os.path.exists("tokenizer.pkl") or not os.path.exists("bi_gru_model.h5"):
    st.error("❌ Model files not found. Make sure 'bi_gru_model.h5' and 'tokenizer.pkl' are in the same directory.")
    st.stop()

# Load tokenizer and model
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

model = tf.keras.models.load_model("bi_gru_model.h5")

# Inference Function
def run_inference(url_only):
    with st.spinner("Analyzing URL for potential threats..."):
        time.sleep(1)  # Simulate processing delay
        input_text = f" {url_only}  "
        sequence = tokenizer.texts_to_sequences([input_text])
        padded = pad_sequences(sequence, maxlen=MAX_LEN, padding="post")
        prediction = model.predict(padded)[0][0]

    st.markdown("### 🔐 Prediction Result")
    if prediction >= 0.8:
        st.error(f"🚨 High Threat Detected – Score: `{prediction:.2f}`")
    elif prediction >= 0.5:
        st.warning(f"⚠️ Moderate Threat Detected – Score: `{prediction:.2f}`")
    else:
        st.success(f"✅ Safe Request – Score: `{prediction:.2f}`")

    st.markdown("---")
    st.caption("Model: Bi-GRU trained on CSIC-2010 dataset (URL-only input for this app).")

# --- Test Case Click Handler ---
st.session_state.setdefault("auto_url", "")
st.session_state.setdefault("auto_trigger", False)

# --- Manual Input Section ---
st.subheader("🧾 Analyze Custom URL")

user_url = st.text_input("Paste the full URL to analyze", value=st.session_state.auto_url)
check_btn = st.button("Check for Threat")

# Trigger inference automatically after button-click example
if check_btn or st.session_state.auto_trigger:
    if user_url.strip() == "":
        st.warning("Please enter a URL before submitting.")
    else:
        run_inference(user_url)
    st.session_state.auto_trigger = False

# --- Mixed Example URLs ---
st.subheader("🧪 Example URLs")
example_urls = {
    "https://silver-lamp-v6v9j59v4w7v2xp7j-8501.app.github.dev/#analyze-custom-request",
    "https://stackoverflow.com/questions/678236/how-to-get-the-filename-without-the-extension-from-a-path-in-python",
    "https://github.com/openai/gym/tree/master/gym/envs",
    "/login.php?username=admin'--&password=1234",
    "/search.php?query=<script>alert('XSS')</script>",
    "/index.php?page=../../../../etc/passwd",
    "/ping?ip=127.0.0.1; reboot"
}

for url in example_urls:
    if st.button(f"Run: {url}"):
        st.session_state.auto_url = url
        st.session_state.auto_trigger = True
