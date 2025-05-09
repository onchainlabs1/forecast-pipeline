#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import json

def print_json(data):
    """Print JSON data in a readable format."""
    print(json.dumps(data, indent=2))

def get_token(base_url):
    """Obtém um token de autenticação."""
    try:
        auth_data = {
            "username": "johndoe",
            "password": "secret"
        }
        response = requests.post(f"{base_url}/token", data=auth_data)
        if response.status_code == 200:
            token_data = response.json()
            return token_data.get("access_token")
        else:
            print(f"Falha na autenticação: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"Erro ao obter token: {e}")
        return None

def main():
    print("Testando a API de previsão de vendas...")
    print("-----------------------------------------")
    
    base_url = "http://localhost:8000"
    
    # Obter token
    token = get_token(base_url)
    headers = {}
    if token:
        print(f"Token obtido: {token[:10]}...")
        headers = {"Authorization": f"Bearer {token}"}
    else:
        print("Token não obtido, tentando requisições sem autenticação...")
    
    # 1. Teste de health check
    print("\n1. Teste de health check:")
    response = requests.get(f"{base_url}/health")
    health_data = response.json()
    print_json(health_data)
    print("-----------------------------------------")
    
    # 2. Teste de previsão individual
    print("\n2. Teste de previsão individual:")
    prediction_params = {
        "store_nbr": 1,
        "family": "PRODUCE",
        "onpromotion": False,
        "date": "2025-05-10"
    }
    response = requests.get(f"{base_url}/predict_single", params=prediction_params, headers=headers)
    try:
        prediction_data = response.json()
        print_json(prediction_data)
        print(f"Status: {response.status_code}")
    except Exception as e:
        print(f"Erro ao processar resposta: {e}")
        print(f"Resposta: {response.text}")
    print("-----------------------------------------")
    
    # 3. Teste de explicação
    print("\n3. Teste de explicação:")
    prediction_id = "1-PRODUCE-2025-05-10"
    explanation_params = {
        "store_nbr": 1,
        "family": "PRODUCE",
        "onpromotion": False,
        "date": "2025-05-10"
    }
    response = requests.get(f"{base_url}/explain/{prediction_id}", params=explanation_params, headers=headers)
    try:
        explanation_data = response.json()
        print_json(explanation_data)
        print(f"Status: {response.status_code}")
        
        # Verificar se a explicação contém feature_contributions
        if "feature_contributions" in explanation_data:
            print(f"\nNúmero de características explicadas: {len(explanation_data['feature_contributions'])}")
            print("\nPrincipais contribuições:")
            top_features = explanation_data["feature_contributions"][:5]
            for feature in top_features:
                print(f"  - {feature['feature']}: {feature['contribution']:.4f}")
    except Exception as e:
        print(f"Erro ao processar resposta: {e}")
        print(f"Resposta: {response.text}")
    print("-----------------------------------------")
    
    print("Testes concluídos.")

if __name__ == "__main__":
    main() 