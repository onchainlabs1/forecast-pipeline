#!/bin/bash

echo "Iniciando testes da API de previsão de vendas"
echo "============================================="

# Teste de previsão
echo "Executando teste de previsão..."
python test_prediction.py > prediction_results.txt 2>&1
echo "Resultado salvo em prediction_results.txt"

# Teste de métricas
echo "Executando teste de métricas..."
python test_metrics.py > metrics_results.txt 2>&1
echo "Resultado salvo em metrics_results.txt"

echo "Testes concluídos."
echo "Verifique os arquivos prediction_results.txt e metrics_results.txt para os resultados." 