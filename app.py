import streamlit as st
import pickle
import re

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🧠",
    layout="centered"
)

# ------------------ NEON STYLE ------------------
st.markdown("""
<style>
body {
    background-color: #0b0f1a;
}
.stApp {
    background: radial-gradient(circle at top, #0b0f1a, #000000);
    color: white;
}

/* Neon Title */
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

/* Subtitle */
.neon-subtitle {
    text-align: center;
    color: #00f0ff;
    text-shadow: 0 0 10px #00f0ff;
}

/* Button Styling */
div.stButton > button {
    background: linear-gradient(90deg, #ff00ff, #00f0ff);
    color: white;
    border-radius: 25px;
    padding: 10px 20px;
    font-size: 16px;
    border: none;
    box-shadow: 0 0 10px #00f0ff;
}

/* Text Area Styling */
textarea {
    background-color: #111 !important;
    color: white !important;
    border-radius: 10px !important;
}

/* Result Box */
.result-box {
    text-align: center;
    font-size: 30px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ------------------ TITLE ------------------
st.markdown('<div class="neon-title">SENTIMENT ANALYZER</div>', unsafe_allow_html=True)
st.markdown('<div class="neon-subtitle">Analyze Customer Reviews Instantly with AI ⚡</div>', unsafe_allow_html=True)

st.divider()

# ------------------ LOAD MODEL ------------------
model = pickle.load(open("backend/model.pkl", "rb"))
vectorizer = pickle.load(open("backend/vectorizer.pkl", "rb"))

# ------------------ CLEAN FUNCTION ------------------
def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ------------------ INPUT ------------------
user_input = st.text_area("Enter your review here:", height=150)

# ------------------ PREDICTION ------------------
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

        elif result[0] == "negative":
            st.markdown('<div class="result-box" style="color:#ff4b5c;">😡 NEGATIVE</div>', unsafe_allow_html=True)

        else:
            st.markdown('<div class="result-box" style="color:#00f0ff;">😐 NEUTRAL</div>', unsafe_allow_html=True)

# ------------------ FOOTER ------------------
st.divider()
st.caption("Check the emotions in your text!")