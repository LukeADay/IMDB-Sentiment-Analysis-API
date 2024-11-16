from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model and tokenizer
model = load_model("sentiment_model.keras")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Prediction function
def predict_sentiment(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, padding='post', maxlen=200)
    prediction = model.predict(padded_sequence)
    score = float(prediction[0][0])
    label = "Positive" if score >= 0.5 else "Negative"
    return score, label

@app.route("/healthz")
def health_check():
    return "OK", 200

@app.route("/predict", methods=["POST"])  # This should be the /predict route for POST requests
def predict():
    # Parse incoming JSON data from the body
    data = request.get_json()  # Get JSON data sent from the frontend
    text = data.get("text")  # Extract text from JSON

    if text is None:
        return jsonify({"error": "No text provided"}), 400

    score, label = predict_sentiment(text)  # Get sentiment prediction
    return jsonify({"score": score, "label": label})  # Send response back to frontend

if __name__ == "__main__":
    app.run(debug=True)
