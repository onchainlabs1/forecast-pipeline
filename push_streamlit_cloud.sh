#!/bin/bash

# Script para enviar as configurações do Streamlit Cloud para o GitHub
# Uso: ./push_streamlit_cloud.sh [branch]

set -e  # Sair em caso de erro

# Default para "fix/streamlit-cloud-minimal" se nenhuma branch for especificada
BRANCH=${1:-fix/streamlit-cloud-minimal}

echo "Preparando configuração do Streamlit Cloud..."

# Criar uma nova branch se não existir
if ! git rev-parse --verify $BRANCH >/dev/null 2>&1; then
  echo "Criando nova branch: $BRANCH"
  git checkout -b $BRANCH
else
  echo "Usando branch existente: $BRANCH"
  git checkout $BRANCH
fi

# Adicionar os arquivos modificados/criados
git add streamlit_requirements.txt
git add streamlit_cloud_app.py
git add .streamlit/secrets.toml
git add .streamlit/config.toml
git add packages.txt
git add STREAMLIT_CLOUD.md

# Commit das alterações
git commit -m "Adiciona configuração mínima para Streamlit Cloud

- Cria streamlit_cloud_app.py com apenas dependências essenciais
- Adiciona streamlit_requirements.txt sem dependências conflitantes
- Configura .streamlit/secrets.toml para variáveis de ambiente
- Atualiza packages.txt para pacotes de sistema necessários
- Adiciona STREAMLIT_CLOUD.md com instruções detalhadas"

# Push para o GitHub
echo "Enviando alterações para o GitHub..."
git push -u origin $BRANCH

echo "=================================="
echo "Configuração do Streamlit Cloud enviada para $BRANCH"
echo ""
echo "Próximos passos:"
echo "1. Criar um pull request no GitHub"
echo "2. No Streamlit Cloud:"
echo "   - Arquivo principal: streamlit_cloud_app.py"
echo "   - Arquivo de requisitos: streamlit_requirements.txt"
echo "==================================" 