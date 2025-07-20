from flask import Flask, request, jsonify
import numpy as np
import joblib
import mlflow
import flask_monitoringdashboard as dashboard
import os
import logging
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics.pairwise import cosine_similarity
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Diretório base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Configuração de logging: arquivo + console
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "app.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)

# Carrega o TfidfVectorizer treinado
TFIDF_PATH = os.path.join(BASE_DIR, "../notebooks/saved_models/tfidf_vectorizer.pkl")
with open(TFIDF_PATH, "rb") as f:
    tfidf = pickle.load(f)

with open(os.path.join(BASE_DIR, "../notebooks/saved_models/le1_nivel_profissional_vaga.pkl"), "rb") as f:
    le1 = pickle.load(f)
with open(os.path.join(BASE_DIR, "../notebooks/saved_models/le2_nivel_ingles_vaga.pkl"), "rb") as f:
    le2 = pickle.load(f)
with open(os.path.join(BASE_DIR, "../notebooks/saved_models/le3_nivel_ingles.pkl"), "rb") as f:
    le3 = pickle.load(f)
with open(os.path.join(BASE_DIR, "../notebooks/saved_models/le4_nivel_academico.pkl"), "rb") as f:
    le4 = pickle.load(f)

app = Flask(__name__)

# Carrega modelo
MODEL_PATH = os.path.join(BASE_DIR, "../notebooks/saved_models/modelo_datathon.pkl")
model = joblib.load(MODEL_PATH)

# Drift: paths
TRAIN_FEATURES_PATH = os.path.join(BASE_DIR, "../notebooks/saved_models/train_features.csv")
LOGGED_FEATURES_PATH = os.path.join(BASE_DIR, "logged_features.csv")
if os.path.exists(TRAIN_FEATURES_PATH):
    train_features = pd.read_csv(TRAIN_FEATURES_PATH)
else:
    train_features = None
    logging.warning("Arquivo de features de treino não encontrado para drift.")

def log_input_features(features):
    """Salva as features de entrada para monitoramento de drift."""
    df = pd.DataFrame([features], columns=[
        "match_score",
        "nivel_profissional_vaga_enc",
        "nivel_ingles_vaga_enc",
        "nivel_ingles_enc",
        "nivel_academico_enc"
    ])
    file_exists = os.path.isfile(LOGGED_FEATURES_PATH)
    df.to_csv(LOGGED_FEATURES_PATH, mode='a', header=not file_exists, index=False)
    logging.info(f"Features de entrada logadas para drift: {features}")

def check_drift():
    """Compara as features atuais com as do treino usando KS-test e loga no MLflow."""
    if train_features is None or not os.path.isfile(LOGGED_FEATURES_PATH):
        return {"drift": False, "details": "Sem dados suficientes"}
    logged = pd.read_csv(LOGGED_FEATURES_PATH)
    drift_results = {}
    drift_found = False
    with mlflow.start_run(run_name="drift_monitoring", nested=True):
        for col in train_features.columns:
            if col in logged.columns:
                stat, p_value = ks_2samp(train_features[col], logged[col])
                drift = bool(p_value < 0.05) # Considera drift se p < 0.05
                drift_results[col] = {
                    "ks_stat": float(stat),
                    "p_value": float(p_value),
                    "drift": drift
                }
                mlflow.log_metric(f"{col}_ks_stat", float(stat))
                mlflow.log_metric(f"{col}_p_value", float(p_value))
                mlflow.log_metric(f"{col}_drift", int(drift))
                if drift:
                    drift_found = True
                    logging.warning(f"Drift detectado na feature '{col}': p={p_value:.4f}")
                else:
                    logging.info(f"Sem drift na feature '{col}': p={p_value:.4f}")
    if not drift_found:
        return {
            "drift": False,
            "details": "Nenhum drift encontrado nas features monitoradas.",
            "features": drift_results
        }
    else:
        return {
            "drift": True,
            "details": "Drift detectado em pelo menos uma feature.",
            "features": drift_results
        }

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logging.info("Recebida requisição de predição")
        data = request.get_json()
        if not data or 'job_description' not in data or 'cv_text' not in data:
            return jsonify({'error': 'JSON deve conter "job_description" e "cv_text".'}), 400

        desc = data['job_description'] or ""
        cv = data['cv_text'] or ""
        desc_vec = tfidf.transform([desc])
        cv_vec = tfidf.transform([cv])
        match_score = cosine_similarity(desc_vec, cv_vec)[0][0]

        nivel_profissional_vaga = data.get('nivel_profissional_vaga', '')
        nivel_ingles_vaga = data.get('nivel_ingles_vaga', '')
        nivel_ingles = data.get('nivel_ingles', '')
        nivel_academico = data.get('nivel_academico', '')

        nivel_profissional_vaga_enc = le1.transform([nivel_profissional_vaga])[0] if nivel_profissional_vaga in le1.classes_ else 0
        nivel_ingles_vaga_enc = le2.transform([nivel_ingles_vaga])[0] if nivel_ingles_vaga in le2.classes_ else 0
        nivel_ingles_enc = le3.transform([nivel_ingles])[0] if nivel_ingles in le3.classes_ else 0
        nivel_academico_enc = le4.transform([nivel_academico])[0] if nivel_academico in le4.classes_ else 0

        features = [
            match_score,
            nivel_profissional_vaga_enc,
            nivel_ingles_vaga_enc,
            nivel_ingles_enc,
            nivel_academico_enc
        ]
        input_data = np.array(features).reshape(1, -1)

        log_input_features(features)
        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1] if hasattr(model, "predict_proba") else None

        # Log MLflow
        with mlflow.start_run(run_name="inference", nested=True):
            mlflow.log_param("input_length", len(features))
            mlflow.log_metric("prediction", float(pred))
            if prob is not None:
                mlflow.log_metric("prob_contratado", float(prob))
            # Loga o payload de entrada
            mlflow.log_dict(data, "payload.json")

        response = {'prediction': int(pred), 'match_score': float(match_score)}
        if prob is not None:
            response['prob_contratado'] = float(prob)
        logging.info(f"Predição: {response} | Entrada: {features}")
        return jsonify(response)
    except Exception as e:
        logging.exception("Erro na predição")
        return jsonify({'error': str(e)}), 500

@app.route('/drift', methods=['GET'])
def drift():
    results = check_drift()
    logging.info(f"Consulta de drift: {results}")
    return jsonify(results)

dashboard.bind(app)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)