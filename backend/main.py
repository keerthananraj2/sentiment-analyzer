from fastapi import FastAPI
import pickle
import re
from nltk.corpus import stopwords

# Initialize app
app = FastAPI()

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Stopwords
stop_words = set(stopwords.words('english'))

# Cleaning function (same as before)
def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
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

    return {"sentiment": result[0]}