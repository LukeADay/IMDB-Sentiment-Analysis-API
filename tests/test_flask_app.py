import pytest
from flask_app import app

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_sentiment_prediction(client):
    response = client.post('/', data={'review': 'I love this movie!'})
    assert response.status_code == 200
    assert b"Positive" in response.data or b"Negative" in response.data
