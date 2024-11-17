import pytest
from flask_app import app  # Ensure this imports your Flask app correctly

@pytest.fixture
def client():
    # Set up the test client for the Flask app
    with app.test_client() as client:
        yield client

def test_sentiment_prediction(client):
    # Send a POST request to the correct endpoint with JSON data
    response = client.post('/predict', json={'text': 'I love this movie!'})
    
    # Assert the response status code
    assert response.status_code == 200
    
    # Parse the JSON response
    data = response.get_json()
    
    # Assert that the response contains a score and label
    assert 'score' in data
    assert 'label' in data
    
    # Check that the label is either "Positive" or "Negative"
    assert data['label'] in ["Positive", "Negative"]
