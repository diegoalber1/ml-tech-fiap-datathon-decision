# 📈 AI Recruiter

AI Recruiter: Otimizando o Match entre Candidatos e Vagas com Machine Learning.

---

## 🎯 Problema

A Decision enfrenta dificuldade para:

- Padronizar entrevistas e armazenar informações valiosas.
- Identificar engajamento real de candidatos.
- Repetir o padrão de sucesso de candidatos bem alocados.

---

## 💡 Solução Proposta

Desenvolver um sistema de IA híbrido, com:

- Pipeline de Machine Learning supervisionado: para prever a probabilidade de sucesso de um candidato com base em dados históricos.
- Clusterização não supervisionada: para identificar perfis de sucesso (padrões latentes).
- API de prediction, servindo o modelo em produção.
- Docker para empacotamento.
- Monitoramento de drift, log de previsões e dashboard básico.



---

## ⚙️ Pipeline Técnico
### 📂 1. Coleta & Pré-processamento

Base de dados: Base fornecida pela Decision.

Tratamento: Limpeza de valores ausentes, normalização de variáveis numéricas, encoding de variáveis categóricas.

Feature Engineering:

- Geração de features de engajamento (ex: tempo de resposta).
- Extração de palavras-chave de entrevistas transcritas com NLP (se houver).
- Criação de scores de fit cultural.

### 🤖 2. Modelagem
Algoritmo supervisionado: Random Forest.

- Target: candidato_aprovado (1) ou não aprovado (0)  {prediction}
          probabilidade de contracao                  {prob_contratado}

Validação: métricas de precisão, recall e F1-score.

Serialização: pickle.

### 🚀 3. Deployment

API: Flask com rota /predict
- Entrada: JSON com atributos do candidato.
- Saída: Probabilidade de aprovação.

Dockerfile: Empacotamento da API + dependências.

Deploy: Local

### 🧪 4. Testes

Testes unitários:
- Pré-processamento.
- Predição.
- Endpoint da API.

Testes de integração:

- Validação com Postman ou cURL.

### 🔍 5. Monitoramento

- A API expõe um painel interativo em /dashboard com estatísticas de tempo de resposta, chamadas e erros.

- O MLflow é usado para rastrear todas as execuções de treinamento e inferência.


---

## 📁 Estrutura do Projeto

```plaintext
├── api/
├──── app.py # API para predição
├── notebooks/
├──── data/ # Dados para treinamento do modelo
├──── saved_models/ # Modelos e scaler salvos
├──── decision-recruitment-process-model-training.ipynb # Notebook para treino e deploy do modelo
├── requirements.txt
├── Dockerfile
├── start.sh
└── README.md
```

---

## 🚀 Como Rodar Localmente

### 1. Clone o repositório

```bash
git clone https://github.com/diegoalber1/ml-tech-fiap-datathon-decision.git
cd ml-tech-fiap-datathon-decision
```

### 2. Crie e ative o ambiente virtual (opcional)

```bash
python -m venv venv
source venv/bin/activate 
# Windows: venv\Scripts\activate
```
### 3. Instale as dependências

```bash
pip install -r requirements.txt
```
## 📊 Treinar o Modelo

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/decision-recruitment-process-model-training.ipynb
```
## 🔁 Fazer Predições com a API

### 1. Rodar a API localmente

```bash
python api/app.py

mlflow ui --host 0.0.0.0 --port 5001
```
Acesse a API em http://localhost:5000

Dashboard de monitoramento:

http://localhost:5000/dashboard

Mlops UI:

http://localhost:5001

### 2. Exemplo de requisição com curl:

```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{
               "data": [0.02681892, 11.0, 4.0, 0.0, 0.0],
               "feature_names": [
               "match_score",
               "nivel_profissional_vaga_enc",
               "nivel_ingles_vaga_enc",
               "nivel_ingles_enc",
               "nivel_academico_enc"
               ]
          }
```
## 🐳 Deploy com Docker

### 1. Build da imagem

```bash
docker build -t fiap-ai-recruiter .
```
### 2. Run da API via Docker

```bash
docker run -it --rm -p 5000:5000 -p 5001:5001 fiap-ai-recruiter
```



