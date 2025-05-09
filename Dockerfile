FROM python:3.8-slim

WORKDIR /app

# Instale as dependências primeiro
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie o restante do código
COPY . .

# Permita o uso da porta via variável de ambiente ou use 8000 como padrão
ENV PORT=8000

# Configure para ignorar o MLflow em produção
ENV DISABLE_MLFLOW=True

# Comando para iniciar a API
CMD uvicorn src.api.main:app --host 0.0.0.0 --port $PORT 