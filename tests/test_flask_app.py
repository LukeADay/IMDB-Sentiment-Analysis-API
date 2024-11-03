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
    assert b'Please input the movie review' in response.data

def test_sentiment_prediction(client):
    response = client.post('/predict', data={'review_text': 'I love this movie!'})
    assert response.status_code == 200
    assert b'Sentiment score:' in response.data
