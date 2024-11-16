from flask import Flask, request, render_template, jsonify, make_response
from flask_cors import CORS
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Enable CORS globally for all routes (or specify for '/predict' if needed)
CORS(app)  # Allow all origins for /predict route

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

# Health check route
@app.route("/healthz")
def health_check():
    return "OK", 200

# Predict route
@app.route("/predict", methods=["POST", "OPTIONS"])  # Handle both POST and OPTIONS
def predict():
    if request.method == "OPTIONS":  # Handle preflight request
        response = make_response('', 200)
        response.headers['Access-Control-Allow-Origin'] = '*'  # Allow all origins
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'  # Allow POST and OPTIONS
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'  # Allow Content-Type header
        return response
    
    if request.method == "POST":
        text = request.json.get("text")
        if not text:
            return jsonify({"error": "No review text provided"}), 400
        score, label = predict_sentiment(text)
        response = jsonify({"score": score, "label": label})
        response.headers['Access-Control-Allow-Origin'] = '*'  # Allow all origins for the POST response
        return response

if __name__ == "__main__":
    app.run(debug=True)
