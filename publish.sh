#!/bin/bash

# Script para preparar e publicar o código no GitHub

# Configurações
VERSION="1.0.0"
GIT_REPO_URL="https://github.com/seu-usuario/forecast-pipeline-dashboard.git"
BRANCH="main"

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Funções úteis
print_header() {
  echo -e "\n${GREEN}==== $1 ====${NC}\n"
}

print_step() {
  echo -e "${YELLOW}→ $1${NC}"
}

print_error() {
  echo -e "${RED}ERROR: $1${NC}"
}

print_success() {
  echo -e "${GREEN}✓ $1${NC}"
}

# Verificar se Git está instalado
if ! command -v git &> /dev/null; then
  print_error "Git não está instalado. Por favor, instale o Git e tente novamente."
  exit 1
fi

print_header "Publicação do Forecast Pipeline Dashboard v$VERSION"

# Verificar se estamos em um repositório Git
if [ ! -d ".git" ]; then
  print_step "Inicializando repositório Git..."
  git init
  print_success "Repositório Git inicializado"
else
  print_step "Repositório Git já existe"
fi

# Criar diretório de logs se não existir
print_step "Criando diretório de logs..."
mkdir -p logs
touch logs/.gitkeep
print_success "Diretório de logs criado"

# Verificar se há mudanças para commitar
print_step "Verificando mudanças..."
if [[ -n $(git status -s) ]]; then
  print_step "Há mudanças para commitar"
  
  # Adicionar todos os arquivos
  print_step "Adicionando arquivos ao stage..."
  git add .
  
  # Pedir mensagem de commit
  read -p "Digite a mensagem de commit (ou pressione Enter para usar mensagem padrão): " commit_message
  if [ -z "$commit_message" ]; then
    commit_message="Versão $VERSION: Melhorias na explicabilidade do modelo e correções de bugs"
  fi
  
  # Commitar mudanças
  print_step "Commitando mudanças..."
  git commit -m "$commit_message"
  print_success "Mudanças commitadas"
else
  print_step "Não há mudanças para commitar"
fi

# Opção para adicionar tag de versão
read -p "Deseja adicionar uma tag de versão v$VERSION? (s/n): " add_tag
if [ "$add_tag" = "s" ] || [ "$add_tag" = "S" ]; then
  print_step "Adicionando tag v$VERSION..."
  git tag -a "v$VERSION" -m "Versão $VERSION"
  print_success "Tag v$VERSION adicionada"
fi

# Verificar se o remote já existe
if ! git remote | grep -q "origin"; then
  print_step "Adicionando remote 'origin'..."
  read -p "Digite a URL do repositório GitHub (ou pressione Enter para usar $GIT_REPO_URL): " repo_url
  if [ -z "$repo_url" ]; then
    repo_url=$GIT_REPO_URL
  fi
  git remote add origin $repo_url
  print_success "Remote 'origin' adicionado: $repo_url"
else
  print_step "Remote 'origin' já existe"
fi

# Opção para push
read -p "Deseja fazer push para o GitHub agora? (s/n): " do_push
if [ "$do_push" = "s" ] || [ "$do_push" = "S" ]; then
  print_step "Fazendo push para o GitHub..."
  
  # Push da branch
  git push -u origin $BRANCH
  
  # Push das tags se houver
  if [ "$add_tag" = "s" ] || [ "$add_tag" = "S" ]; then
    git push origin --tags
  fi
  
  print_success "Push realizado com sucesso"
  print_success "O código está disponível em: $GIT_REPO_URL"
else
  print_step "Push não realizado. Para fazer push manualmente, execute:"
  echo "  git push -u origin $BRANCH"
  if [ "$add_tag" = "s" ] || [ "$add_tag" = "S" ]; then
    echo "  git push origin --tags"
  fi
fi

print_header "Publicação concluída!"
echo -e "Para iniciar a aplicação, execute: ${YELLOW}bash restart.sh${NC}" 