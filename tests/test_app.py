import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from fastapi.testclient import TestClient
from app import app  # Ensure this imports your FastAPI app correctly

client = TestClient(app)

def test_root():
    response = client.get("/")
    print("Status Code:", response.status_code)
    print("Response Data:", response.json())
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["message"] == "IMDb Sentiment Analysis API is running"

def test_app_starts():
    assert app is not None, "FastAPI app failed to initialize"
