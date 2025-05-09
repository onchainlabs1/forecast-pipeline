FROM python:3.8.10-slim

WORKDIR /app

# Desativar MLflow em produção
ENV DISABLE_MLFLOW=True

# Instale as dependências primeiro
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Crie os diretórios necessários
RUN mkdir -p models reports

# Copie o restante do código
COPY . .

# Comando para iniciar a API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"] 