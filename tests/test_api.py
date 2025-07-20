import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'api')))
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_predict_success(client):
    payload = {
        "data": [0.1, 2, 1, 1, 3]
    }
    response = client.post('/predict', json=payload)
    assert response.status_code == 200
    json_data = response.get_json()
    assert "prediction" in json_data

def test_predict_missing_data(client):
    response = client.post('/predict', json={})
    assert response.status_code == 400
    assert "error" in response.get_json()