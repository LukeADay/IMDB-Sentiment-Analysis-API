import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from sklearn.model_selection import train_test_split
import os
import urllib.request
import tarfile

# Download and extract IMDb dataset
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
dataset_path = "aclImdb_v1.tar.gz"

if not os.path.exists("aclImdb"):
    urllib.request.urlretrieve(url, dataset_path)
    with tarfile.open(dataset_path, "r:gz") as tar:
        tar.extractall()

# Load data
def load_imdb_data(data_dir, label):
    texts = []
    labels = []
    for file_path in os.listdir(data_dir):
        if file_path.endswith(".txt"):
            with open(os.path.join(data_dir, file_path), "r") as file:
                texts.append(file.read())
                labels.append(label)
    return texts, labels

# Prepare positive and negative reviews
train_texts, train_labels = load_imdb_data("aclImdb/train/pos", 1)
neg_texts, neg_labels = load_imdb_data("aclImdb/train/neg", 0)
train_texts += neg_texts
train_labels += neg_labels

# Tokenize and pad text data
vocab_size = 15000
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(train_texts)
X = tokenizer.texts_to_sequences(train_texts)
X = pad_sequences(X, padding='post', maxlen=200)
y = np.array(train_labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128, input_length=200),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(LSTM(32)),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=64)

# Save the trained model and tokenizer
model.save("sentiment_model.keras")
import pickle
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
