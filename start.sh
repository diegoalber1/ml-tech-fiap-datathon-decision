#!/bin/bash
# Inicia o Flask em background
python api/app.py &

# Inicia o MLflow IU de treinamento em background
mlflow ui --host 0.0.0.0 --port 5001 --backend-store-uri "$(pwd)/notebooks/mlruns" &

# Inicia o MLflow UI de inferencia em background
mlflow ui --host 0.0.0.0 --port 5002 --backend-store-uri "$(pwd)/mlruns" &


# Espera qualquer processo terminar (mantém o container vivo)
wait -n