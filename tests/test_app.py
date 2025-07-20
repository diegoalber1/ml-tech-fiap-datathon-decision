
import sys
import os
import pytest
import os
import tempfile
import json
from unittest import mock
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'api')))
from app import app, log_input_features, check_drift

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_predict_success(client):
    # Mocka os encoders e modelo para não depender de arquivos reais
    with mock.patch("app.le1") as le1, \
         mock.patch("app.le2") as le2, \
         mock.patch("app.le3") as le3, \
         mock.patch("app.le4") as le4, \
         mock.patch("app.model") as model, \
         mock.patch("app.tfidf") as tfidf:

        le1.classes_ = ['junior']
        le1.transform.return_value = [1]
        le2.classes_ = ['basico']
        le2.transform.return_value = [2]
        le3.classes_ = ['intermediario']
        le3.transform.return_value = [3]
        le4.classes_ = ['graduado']
        le4.transform.return_value = [4]
        model.predict.return_value = [1]
        model.predict_proba.return_value = [[0.3, 0.7]]
        tfidf.transform.return_value = [[0.1, 0.2]]

        payload = {
            "job_description": "desc",
            "cv_text": "cv",
            "nivel_profissional_vaga": "junior",
            "nivel_ingles_vaga": "basico",
            "nivel_ingles": "intermediario",
            "nivel_academico": "graduado"
        }
        response = client.post('/predict', json=payload)
        assert response.status_code == 200
        data = response.get_json()
        assert "prediction" in data
        assert "match_score" in data
        assert "prob_contratado" in data

def test_predict_missing_fields(client):
    response = client.post('/predict', json={})
    assert response.status_code == 400
    assert "error" in response.get_json()

def test_predict_internal_error(client):
    # Força uma exceção no predict
    with mock.patch("app.tfidf") as tfidf:
        tfidf.transform.side_effect = Exception("Erro de teste")
        payload = {
            "job_description": "desc",
            "cv_text": "cv"
        }
        response = client.post('/predict', json=payload)
        assert response.status_code == 500
        assert "error" in response.get_json()

def test_drift_no_data(client):
    # Mocka para simular ausência de dados de drift
    with mock.patch("app.train_features", None):
        response = client.get('/drift')
        assert response.status_code == 200
        data = response.get_json()
        assert data["drift"] is False

def test_log_input_features_creates_file(tmp_path):
    # Testa se a função cria o arquivo corretamente
    test_file = tmp_path / "logged_features.csv"
    features = [0.1, 1, 2, 3, 4]
    with mock.patch("app.LOGGED_FEATURES_PATH", str(test_file)):
        log_input_features(features)
        assert os.path.exists(test_file)
