from flask import Flask, request, render_template, jsonify
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

@app.route("/", methods=["GET", "POST"])
def home():
    print(f"Request method: {request.method}")
    print(f"Request content-type: {request.content_type}")
    if request.method == "POST":
        text = request.form.get("review")
        print(f"Received review: {text}")
        score, label = predict_sentiment(text)
        return jsonify({"review": text, "score": score, "label": label})
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)

