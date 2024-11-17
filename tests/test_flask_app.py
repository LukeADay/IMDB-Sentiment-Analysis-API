import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from flask_app import app  # Ensure this imports your Flask app correctly

@pytest.fixture
def client():
    # Set up the test client for the Flask app
    with app.test_client() as client:
        yield client

def test_sentiment_prediction(client):
    response = client.post('/predict', json={'text': 'I love this movie!'})
    print("Status Code:", response.status_code)
    print("Response Data:", response.get_json())
    assert response.status_code == 200
    data = response.get_json()
    assert 'score' in data
    assert 'label' in data
    assert data['label'] in ["Positive", "Negative"]

def test_app_starts(client):
    assert client is not None, "Flask test client failed to initialize"
