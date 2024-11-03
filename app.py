from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the model and tokenizer
model = load_model("sentiment_model.keras")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.get("/")
async def root():
    return {"message": "IMDb Sentiment Analysis API is running"}

@app.post("/predict")
async def predict_sentiment(input: TextInput):
    # Preprocess text
    sequence = tokenizer.texts_to_sequences([input.text])
    padded_sequence = pad_sequences(sequence, padding='post', maxlen=200)
    
    # Predict sentiment
    prediction = model.predict(padded_sequence)
    label = "Positive" if prediction[0][0] >= 0.5 else "Negative"
    score = float(prediction[0][0])
    
    return {"label": label, "score": score}
