from fastapi import FastAPI
import pickle
import re
import os
import nltk
from nltk.corpus import stopwords

# Ensure stopwords are available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# Initialize app
app = FastAPI()

# Safe path loading (important for deployment)
BASE_DIR = os.path.dirname(__file__)

model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb"))

# Cleaning function (must match training)
def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r'http\S+', '', text)   # remove URLs
    text = re.sub(r'@\w+', '', text)      # remove mentions
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# Home route
@app.get("/")
def home():
    return {"message": "Sentiment Analyzer API is running 🚀"}

# Prediction route
@app.post("/predict")
def predict(review: str):
    cleaned = clean_text(review)
    vec = vectorizer.transform([cleaned])
    result = model.predict(vec)

    return {
        "review": review,
        "sentiment": result[0]
    }