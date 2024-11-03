import pytest
from flask_app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_page(client):
    response = client.get('/')
    assert response.status_code == 200

def test_sentiment_prediction(client):
    # Simulate a POST request to the home route with form data
    response = client.post('/', data={'review': 'I love this movie!'})
    assert response.status_code == 200
    assert b"Positive" in response.data or b"Negative" in response.data  # Check for either label in response
