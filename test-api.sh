#!/bin/bash

echo "Testing the Sales Forecasting API..."
echo "-----------------------------------------"

# Get an access token
echo "Authenticating..."
TOKEN_RESPONSE=$(curl -s -X POST -d "username=johndoe&password=secret" http://localhost:8000/token)
echo "Token response: $TOKEN_RESPONSE"
TOKEN=$(echo $TOKEN_RESPONSE | grep -o '"access_token":"[^"]*' | cut -d'"' -f4)

if [ -z "$TOKEN" ]; then
    echo "Could not obtain token. Using direct approach..."
    # Test individual prediction without token
    echo "Testing individual prediction for Store 1, PRODUCE, 2025-05-10..."
    PREDICTION_RESPONSE=$(curl -s -X GET "http://localhost:8000/predict_single?store_nbr=1&family=PRODUCE&onpromotion=false&date=2025-05-10")
    echo "$PREDICTION_RESPONSE" | cat
    echo "-----------------------------------------"
    
    # Test prediction explanation without token
    echo "Testing prediction explanation..."
    EXPLANATION_RESPONSE=$(curl -s -X GET "http://localhost:8000/explain/1-PRODUCE-2025-05-10?store_nbr=1&family=PRODUCE&onpromotion=false&date=2025-05-10")
    echo "$EXPLANATION_RESPONSE" | cat
else
    echo "Token: $TOKEN"
    echo "-----------------------------------------"
    
    # Test individual prediction with token
    echo "Testing individual prediction for Store 1, PRODUCE, 2025-05-10..."
    PREDICTION_RESPONSE=$(curl -s -X GET -H "Authorization: Bearer $TOKEN" "http://localhost:8000/predict_single?store_nbr=1&family=PRODUCE&onpromotion=false&date=2025-05-10")
    echo "$PREDICTION_RESPONSE" | cat
    echo "-----------------------------------------"
    
    # Test prediction explanation with token
    echo "Testing prediction explanation..."
    EXPLANATION_RESPONSE=$(curl -s -X GET -H "Authorization: Bearer $TOKEN" "http://localhost:8000/explain/1-PRODUCE-2025-05-10?store_nbr=1&family=PRODUCE&onpromotion=false&date=2025-05-10")
    echo "$EXPLANATION_RESPONSE" | cat
fi

echo "-----------------------------------------"
echo "Tests completed."
