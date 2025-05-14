#!/bin/bash

# Script para versionar e subir o projeto ao GitHub
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
VERSION="1.0.0"
BRANCH="main"
REPO_URL="https://github.com/onchainlabs1/forecast-pipeline.git"

echo "===== Preparando o Forecast Pipeline Dashboard para GitHub ====="
echo "Repositório: $REPO_URL"
echo "Branch: $BRANCH"
echo "Versão: $VERSION"

# Verificar se está sendo executado com bash
if [ -z "$BASH_VERSION" ]; then
  echo "Este script deve ser executado com bash. Use:"
  echo "bash publish_github.sh"
  exit 1
fi

# Adicionar arquivos ao Git
echo "Adicionando arquivos ao controle de versão..."
git add .

# Verificar status
echo "Status do repositório:"
git status

# Criar commit
echo "Criando commit com versão $VERSION..."
git commit -m "Forecast Pipeline Dashboard v$VERSION - $TIMESTAMP"

# Enviar para GitHub
echo "Enviando para GitHub..."
git push origin $BRANCH || echo "Falha ao enviar para o GitHub. Tente manualmente: git push origin $BRANCH"

echo ""
echo "===== Processo concluído! ====="
echo "Para verificar o repositório no GitHub acesse:"
echo "$REPO_URL" 