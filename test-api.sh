#!/bin/bash

echo "Testando a API de previsão de vendas..."
echo "-----------------------------------------"

# Obter um token de acesso
echo "Autenticando..."
TOKEN_RESPONSE=$(curl -s -X POST -d "username=johndoe&password=secret" http://localhost:8000/token)
echo "Resposta de token: $TOKEN_RESPONSE"
TOKEN=$(echo $TOKEN_RESPONSE | grep -o '"access_token":"[^"]*' | cut -d'"' -f4)

if [ -z "$TOKEN" ]; then
    echo "Não foi possível obter o token. Usando abordagem direta..."
    # Testar previsão individual sem token
    echo "Testando previsão individual para Store 1, PRODUCE, 2025-05-10..."
    PREDICTION_RESPONSE=$(curl -s -X GET "http://localhost:8000/predict_single?store_nbr=1&family=PRODUCE&onpromotion=false&date=2025-05-10")
    echo "$PREDICTION_RESPONSE" | cat
    echo "-----------------------------------------"
    
    # Testar explicação da previsão sem token
    echo "Testando explicação da previsão..."
    EXPLANATION_RESPONSE=$(curl -s -X GET "http://localhost:8000/explain/1-PRODUCE-2025-05-10?store_nbr=1&family=PRODUCE&onpromotion=false&date=2025-05-10")
    echo "$EXPLANATION_RESPONSE" | cat
else
    echo "Token: $TOKEN"
    echo "-----------------------------------------"
    
    # Testar previsão individual com token
    echo "Testando previsão individual para Store 1, PRODUCE, 2025-05-10..."
    PREDICTION_RESPONSE=$(curl -s -X GET -H "Authorization: Bearer $TOKEN" "http://localhost:8000/predict_single?store_nbr=1&family=PRODUCE&onpromotion=false&date=2025-05-10")
    echo "$PREDICTION_RESPONSE" | cat
    echo "-----------------------------------------"
    
    # Testar explicação da previsão com token
    echo "Testando explicação da previsão..."
    EXPLANATION_RESPONSE=$(curl -s -X GET -H "Authorization: Bearer $TOKEN" "http://localhost:8000/explain/1-PRODUCE-2025-05-10?store_nbr=1&family=PRODUCE&onpromotion=false&date=2025-05-10")
    echo "$EXPLANATION_RESPONSE" | cat
fi

echo "-----------------------------------------"
echo "Testes concluídos."
