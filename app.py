import streamlit as st
import pickle
import re
import os
import nltk
from nltk.corpus import stopwords

# Download stopwords (needed for deployment)
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# Page config
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🧠",
    layout="centered"
)

# Styling
st.markdown("""
<style>
body {
    background-color: #0b0f1a;
}
.stApp {
    background: radial-gradient(circle at top, #0b0f1a, #000000);
    color: white;
}

.neon-title {
    text-align: center;
    font-size: 60px;
    font-weight: bold;
    color: #fff;
    text-shadow:
        0 0 5px #00f0ff,
        0 0 10px #00f0ff,
        0 0 20px #00f0ff,
        0 0 40px #ff00ff,
        0 0 80px #ff00ff;
}

.neon-subtitle {
    text-align: center;
    color: #00f0ff;
    text-shadow: 0 0 10px #00f0ff;
}

div.stButton > button {
    background: linear-gradient(90deg, #ff00ff, #00f0ff);
    color: white;
    border-radius: 25px;
    padding: 10px 20px;
    font-size: 16px;
    border: none;
    box-shadow: 0 0 10px #00f0ff;
}

textarea {
    background-color: #111 !important;
    color: white !important;
    border-radius: 10px !important;
}

.result-box {
    text-align: center;
    font-size: 30px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="neon-title">SENTIMENT ANALYZER</div>', unsafe_allow_html=True)
st.markdown('<div class="neon-subtitle">Analyze Customer Reviews Instantly with AI ⚡</div>', unsafe_allow_html=True)

st.divider()

# Safe model loading (important for deployment)
BASE_DIR = os.path.dirname(__file__)

model = pickle.load(open(os.path.join(BASE_DIR, "backend/model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(BASE_DIR, "backend/vectorizer.pkl"), "rb"))

# Correct cleaning (same as training)
def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r'http\S+', '', text)   # remove URLs
    text = re.sub(r'@\w+', '', text)      # remove mentions
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# Input
user_input = st.text_area("Enter your review here:", height=150)

# Prediction
if st.button("🔍 Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a review!")
    else:
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        result = model.predict(vec)

        st.divider()

        if result[0] == "positive":
            st.markdown('<div class="result-box" style="color:#00ff88;">😊 POSITIVE</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-box" style="color:#ff4b5c;">😡 NEGATIVE</div>', unsafe_allow_html=True)

st.divider()
st.caption("Check the emotions in your text!")