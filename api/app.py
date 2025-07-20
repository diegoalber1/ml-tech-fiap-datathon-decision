from flask import Flask, request, jsonify
import numpy as np
import joblib
import mlflow
import flask_monitoringdashboard as dashboard
import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)

# Carrega modelo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../notebooks/saved_models/modelo_datathon.pkl")
model = joblib.load(MODEL_PATH)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'data' not in data:
            return jsonify({'error': 'JSON com chave "data" é obrigatório.'}), 400

        # Espera lista de features no mesmo formato do treino
        input_data = np.array(data['data']).reshape(1, -1)

        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1] if hasattr(model, "predict_proba") else None

        # Log MLflow
        with mlflow.start_run(run_name="inference", nested=True):
            mlflow.log_param("input_length", len(data['data']))
            mlflow.log_metric("prediction", float(pred))
            if prob is not None:
                mlflow.log_metric("prob_contratado", float(prob))

        response = {'prediction': int(pred)}
        if prob is not None:
            response['prob_contratado'] = float(prob)
        return jsonify(response)

    except Exception as e:
        logging.exception("Erro na predição")
        return jsonify({'error': str(e)}), 500

dashboard.bind(app)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)