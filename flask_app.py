from flask import Flask, request, render_template, jsonify
from flask_cors import CORS  # Import CORS
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})  # Enable CORS for the /predict route specifically

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

@app.route("/predict", methods=["POST", "OPTIONS"])  # Allow OPTIONS method explicitly
def predict():
    if request.method == "OPTIONS":  # CORS preflight request
        return '', 200  # Respond with a successful status for OPTIONS
    if request.method == "POST":
        text = request.json.get("text")
        if not text:
            return jsonify({"error": "No review text provided"}), 400
        score, label = predict_sentiment(text)
        return jsonify({"score": score, "label": label})

if __name__ == "__main__":
    app.run(debug=True)
