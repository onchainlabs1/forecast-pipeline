#!/bin/bash

# Script para criar o arquivo kaggle.json

# Cria o diretório ~/.kaggle se não existir
mkdir -p ~/.kaggle

# Pede as informações do usuário
echo "Digite seu nome de usuário do Kaggle:"
read username

echo "Digite sua chave de API do Kaggle:"
read key

# Cria o arquivo kaggle.json
echo "{
  \"username\": \"$username\",
  \"key\": \"$key\"
}" > ~/.kaggle/kaggle.json

# Define as permissões corretas
chmod 600 ~/.kaggle/kaggle.json

echo "Arquivo kaggle.json criado com sucesso em ~/.kaggle/kaggle.json"
echo "Testando a configuração..."

# Testa a configuração
kaggle competitions list | head -n 5

echo ""
echo "Se você viu uma lista de competições acima, a configuração está correta!"
echo "Agora você pode executar: python3 src/data/load_data.py" 