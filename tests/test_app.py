from fastapi.testclient import TestClient
from app import app  # Ensure this imports your FastAPI app correctly

client = TestClient(app)

def test_root():
    # Test the root (/) endpoint
    response = client.get("/")
    
    # Assert the response status code
    assert response.status_code == 200
    
    # Assert the response contains a JSON object with a "message" key
    data = response.json()
    assert "message" in data
    assert data["message"] == "Hello, World!"  # Adjust this based on your actual root response

def test_sentiment_prediction():
    # Test the /predict endpoint with JSON input
    response = client.post('/predict', json={'text': 'I love this movie!'})
    
    # Assert the response status code
    assert response.status_code == 200
    
    # Assert the response contains a JSON object with "score" and "label"
    data = response.json()
    assert 'score' in data
    assert 'label' in data
    
    # Check that the label is either "Positive" or "Negative"
    assert data['label'] in ["Positive", "Negative"]
