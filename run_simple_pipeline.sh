#!/bin/bash

# Script simplificado para executar o pipeline

echo "Iniciando pipeline simplificado..."

# 1. Instalar dependências necessárias
echo "Instalando dependências..."
pip3 install pandas numpy scikit-learn lightgbm matplotlib seaborn prophet statsmodels jupyter mlflow 

# 2. Pré-processamento dos dados
echo "Pré-processando dados..."
python3 src/data/preprocess.py

# 3. Treinamento do modelo
echo "Treinando modelo..."
python3 src/train_model.py

echo "Pipeline completo!" 