#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime, timedelta
import random
from src.database.database import db_session
from src.database.models import ModelDrift
from src.database.repository import ModelDriftRepository

def main():
    """Adiciona dados de exemplo para model drift."""
    print("Adicionando dados de exemplo para model drift...")
    
    # Modelo para o qual queremos adicionar drift data
    model_name = "LightGBM (Production)"
    model_version = "1.0.0"
    
    # Métricas a serem salvas
    metrics = ["rmse", "mae"]
    
    # Gerar dados para os últimos 14 dias
    end_date = datetime.utcnow()
    
    # Listas para os dados
    drift_data_list = []
    
    # Base values
    base_rmse = 150.0
    base_mae = 85.0
    
    # Generate data for each day
    for day in range(14, 0, -1):
        # Data para este registro
        current_date = end_date - timedelta(days=day)
        
        # Adicionar variação aleatória com tendência de degradação
        degradation_factor = day / 14.0  # Quanto mais recente (menor day), maior a degradação
        random_factor = random.uniform(0.8, 1.2)
        
        # RMSE
        rmse_value = base_rmse * (1 + (1 - degradation_factor) * 0.3) * random_factor
        exceeded_rmse = rmse_value > (base_rmse * 1.2)  # Threshold é 20% acima do valor base
        
        # Adicionar registro RMSE
        drift_data_list.append({
            "model_name": model_name,
            "model_version": model_version,
            "drift_metric": "rmse",
            "drift_value": rmse_value,
            "threshold": base_rmse * 1.2,
            "exceeded_threshold": exceeded_rmse,
            "timestamp": current_date
        })
        
        # MAE
        mae_value = base_mae * (1 + (1 - degradation_factor) * 0.3) * random_factor
        exceeded_mae = mae_value > (base_mae * 1.2)  # Threshold é 20% acima do valor base
        
        # Adicionar registro MAE
        drift_data_list.append({
            "model_name": model_name,
            "model_version": model_version,
            "drift_metric": "mae",
            "drift_value": mae_value,
            "threshold": base_mae * 1.2,
            "exceeded_threshold": exceeded_mae,
            "timestamp": current_date
        })
    
    # Salvar todos os registros no banco de dados
    with db_session() as db:
        # Limpar registros existentes (opcional)
        db.query(ModelDrift).filter(
            ModelDrift.model_name == model_name,
            ModelDrift.model_version == model_version
        ).delete()
        db.commit()
        
        # Adicionar novos registros
        ModelDriftRepository.create_many(db, drift_data_list)
        
        print(f"Adicionados {len(drift_data_list)} registros de model drift.")
    
    print("\nOperação concluída!")

if __name__ == "__main__":
    main() 