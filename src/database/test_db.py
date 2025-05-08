#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para testar as operações do banco de dados.
"""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime, timedelta
import random
import numpy as np

# Adiciona o diretório raiz ao path
project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root))

from src.database.database import db_session
from src.database.repository import (
    StoreRepository,
    ProductFamilyRepository,
    HistoricalSalesRepository,
    PredictionRepository,
    ModelMetricRepository,
    FeatureImportanceRepository
)

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def test_add_prediction():
    """Teste para adicionar uma previsão ao banco de dados."""
    logger.info("Testando adição de previsão ao banco de dados")
    
    with db_session() as session:
        # Obter uma loja e família
        store = StoreRepository.get_by_store_nbr(session, 1)
        family = ProductFamilyRepository.get_by_name(session, "GROCERY I")
        
        if not store or not family:
            logger.error("Loja ou família não encontrada")
            return
        
        # Criar dados de previsão
        prediction_date = datetime.now()
        target_date = prediction_date + timedelta(days=1)
        
        prediction_data = {
            "store_id": store.id,
            "family_id": family.id,
            "prediction_date": prediction_date,
            "target_date": target_date,
            "onpromotion": False,
            "predicted_sales": 250.75,
            "prediction_interval_lower": 230.5,
            "prediction_interval_upper": 270.0,
            "model_version": "1.0.0",
            "feature_values": {"day_of_week": 2, "is_weekend": 0, "onpromotion": 0}
        }
        
        # Salvar previsão
        prediction = PredictionRepository.create(session, prediction_data)
        logger.info(f"Previsão criada com ID: {prediction.id}")
        
        # Recuperar previsão
        saved_prediction = PredictionRepository.get_by_id(session, prediction.id)
        logger.info(f"Previsão recuperada: {saved_prediction}")
        
        return saved_prediction.id


def test_add_metrics():
    """Teste para adicionar métricas de modelo ao banco de dados."""
    logger.info("Testando adição de métricas ao banco de dados")
    
    with db_session() as session:
        # Criar dados de métricas
        metrics_data = [
            {
                "model_name": "store-sales-forecaster",
                "model_version": "1.0.0",
                "metric_name": "rmse",
                "metric_value": 0.45,
                "timestamp": datetime.now()
            },
            {
                "model_name": "store-sales-forecaster",
                "model_version": "1.0.0",
                "metric_name": "mae",
                "metric_value": 0.32,
                "timestamp": datetime.now()
            },
            {
                "model_name": "store-sales-forecaster",
                "model_version": "1.0.0",
                "metric_name": "mape",
                "metric_value": 12.5,
                "timestamp": datetime.now()
            },
            {
                "model_name": "store-sales-forecaster",
                "model_version": "1.0.0",
                "metric_name": "r2",
                "metric_value": 0.87,
                "timestamp": datetime.now()
            }
        ]
        
        # Salvar métricas
        metrics = ModelMetricRepository.create_many(session, metrics_data)
        logger.info(f"Métricas criadas: {len(metrics)}")
        
        # Recuperar métricas
        latest_metrics = ModelMetricRepository.get_latest_metrics(
            session, "store-sales-forecaster"
        )
        logger.info(f"Métricas recuperadas: {latest_metrics}")
        
        return latest_metrics


def test_add_feature_importance():
    """Teste para adicionar importância de atributos ao banco de dados."""
    logger.info("Testando adição de importância de atributos ao banco de dados")
    
    with db_session() as session:
        # Criar dados de importância de atributos
        features = [
            "day_of_week", "month", "day_of_month", "onpromotion",
            "is_weekend", "is_month_start", "is_month_end",
            "store_nbr", "family_GROCERY I", "family_BEVERAGES"
        ]
        
        importance_data_list = []
        total_importance = 0
        
        # Gerar valores aleatórios de importância
        raw_values = [random.random() for _ in range(len(features))]
        total = sum(raw_values)
        normalized_values = [value / total for value in raw_values]
        
        # Criar registros de importância
        for i, feature in enumerate(features):
            importance_data_list.append({
                "model_name": "store-sales-forecaster",
                "model_version": "1.0.0",
                "feature_name": feature,
                "importance_value": normalized_values[i],
                "timestamp": datetime.now()
            })
        
        # Salvar importância de atributos
        importance_records = FeatureImportanceRepository.create_many(
            session, importance_data_list
        )
        logger.info(f"Registros de importância criados: {len(importance_records)}")
        
        # Recuperar importância de atributos
        importance_df = FeatureImportanceRepository.get_feature_importance_as_dataframe(
            session, "store-sales-forecaster", "1.0.0"
        )
        logger.info(f"Registros de importância recuperados: {len(importance_df)}")
        
        return importance_df


def test_query_historical_sales():
    """Teste para consultar vendas históricas do banco de dados."""
    logger.info("Testando consulta de vendas históricas do banco de dados")
    
    with db_session() as session:
        # Obter uma loja e família
        store = StoreRepository.get_by_store_nbr(session, 1)
        family = ProductFamilyRepository.get_by_name(session, "GROCERY I")
        
        if not store or not family:
            logger.error("Loja ou família não encontrada")
            return
        
        # Consultar histórico de vendas
        sales_df = HistoricalSalesRepository.get_sales_history_as_dataframe(
            session, store_id=store.id, family_id=family.id, days=30
        )
        
        logger.info(f"Vendas históricas recuperadas: {len(sales_df)}")
        if not sales_df.empty:
            logger.info(f"Primeiras linhas:\n{sales_df.head()}")
        
        return sales_df


def test_add_more_historical_sales():
    """Teste para adicionar mais vendas históricas ao banco de dados."""
    logger.info("Testando adição de mais vendas históricas ao banco de dados")
    
    with db_session() as session:
        # Obter lojas e famílias
        stores = StoreRepository.get_all(session)
        families = ProductFamilyRepository.get_all(session)
        
        if not stores or not families:
            logger.error("Lojas ou famílias não encontradas")
            return
        
        # Limitar para 5 lojas e 3 famílias para não criar dados demais
        stores = stores[:5]
        families = families[:3]
        
        # Gerar vendas históricas para os últimos 30 dias
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=30)
        
        total_records = 0
        
        # Fatores de sazonalidade
        day_of_week_factors = {0: 0.8, 1: 0.9, 2: 1.0, 3: 1.0, 4: 1.2, 5: 1.5, 6: 0.7}  # Seg-Dom
        
        for store in stores:
            for family in families:
                sales_data_list = []
                
                for days_ago in range(30, 0, -1):
                    current_date = end_date - timedelta(days=days_ago)
                    current_datetime = datetime.combine(current_date, datetime.min.time())
                    
                    # Fator de dia da semana
                    day_factor = day_of_week_factors[current_date.weekday()]
                    
                    # Fator mensal (efeito sazonal)
                    month_factor = 1.0 + 0.1 * np.sin(2 * np.pi * current_date.month / 12)
                    
                    # Fator da loja (lojas diferentes têm volumes de vendas diferentes)
                    store_factor = 0.5 + 1.5 * store.id / len(stores)
                    
                    # Fator da família (produtos diferentes têm volumes de vendas diferentes)
                    family_factor = 0.5 + 1.5 * family.id / len(families)
                    
                    # Promoção aleatória
                    onpromotion = random.random() < 0.2
                    promotion_factor = 1.3 if onpromotion else 1.0
                    
                    # Vendas base com sazonalidade e fatores
                    base_sales = 100 * day_factor * month_factor * store_factor * family_factor * promotion_factor
                    
                    # Adicionar ruído aleatório
                    noise = random.uniform(0.8, 1.2)
                    sales = base_sales * noise
                    
                    # Criar dados de vendas
                    sales_data = {
                        "store_id": store.id,
                        "family_id": family.id,
                        "date": current_datetime,
                        "sales": sales,
                        "onpromotion": onpromotion
                    }
                    
                    sales_data_list.append(sales_data)
                
                # Inserir lote
                if sales_data_list:
                    HistoricalSalesRepository.create_many(session, sales_data_list)
                    total_records += len(sales_data_list)
                    logger.info(f"Inseridos {len(sales_data_list)} registros históricos para loja {store.store_nbr}, família {family.name}")
        
        logger.info(f"Total de registros históricos inseridos: {total_records}")
        return total_records


def test_add_future_predictions():
    """Teste para adicionar previsões futuras ao banco de dados."""
    logger.info("Testando adição de previsões futuras ao banco de dados")
    
    with db_session() as session:
        # Obter lojas e famílias
        stores = StoreRepository.get_all(session)
        families = ProductFamilyRepository.get_all(session)
        
        if not stores or not families:
            logger.error("Lojas ou famílias não encontradas")
            return
        
        # Limitar para 5 lojas e 3 famílias para não criar dados demais
        stores = stores[:5]
        families = families[:3]
        
        # Gerar previsões para os próximos 14 dias
        prediction_date = datetime.now()
        
        total_records = 0
        
        # Fatores de sazonalidade
        day_of_week_factors = {0: 0.8, 1: 0.9, 2: 1.0, 3: 1.0, 4: 1.2, 5: 1.5, 6: 0.7}  # Seg-Dom
        
        for store in stores:
            for family in families:
                prediction_data_list = []
                
                for days_ahead in range(1, 15):
                    target_date = prediction_date + timedelta(days=days_ahead)
                    
                    # Fator de dia da semana
                    day_factor = day_of_week_factors[target_date.weekday()]
                    
                    # Fator mensal (efeito sazonal)
                    month_factor = 1.0 + 0.1 * np.sin(2 * np.pi * target_date.month / 12)
                    
                    # Fator da loja (lojas diferentes têm volumes de vendas diferentes)
                    store_factor = 0.5 + 1.5 * store.id / len(stores)
                    
                    # Fator da família (produtos diferentes têm volumes de vendas diferentes)
                    family_factor = 0.5 + 1.5 * family.id / len(families)
                    
                    # Promoção aleatória
                    onpromotion = random.random() < 0.2
                    promotion_factor = 1.3 if onpromotion else 1.0
                    
                    # Vendas base com sazonalidade e fatores
                    base_sales = 100 * day_factor * month_factor * store_factor * family_factor * promotion_factor
                    
                    # Adicionar ruído aleatório
                    noise = random.uniform(0.8, 1.2)
                    predicted_sales = base_sales * noise
                    
                    # Calcular intervalos de previsão
                    lower_bound = predicted_sales * 0.9
                    upper_bound = predicted_sales * 1.1
                    
                    # Criar dados de previsão
                    prediction_data = {
                        "store_id": store.id,
                        "family_id": family.id,
                        "prediction_date": prediction_date,
                        "target_date": target_date,
                        "onpromotion": onpromotion,
                        "predicted_sales": predicted_sales,
                        "prediction_interval_lower": lower_bound,
                        "prediction_interval_upper": upper_bound,
                        "model_version": "1.0.0",
                        "feature_values": {
                            "day_of_week": target_date.weekday(),
                            "is_weekend": 1 if target_date.weekday() >= 5 else 0,
                            "onpromotion": 1 if onpromotion else 0
                        }
                    }
                    
                    # Salvar previsão
                    prediction = PredictionRepository.create(session, prediction_data)
                    total_records += 1
                
                logger.info(f"Inseridas {days_ahead} previsões para loja {store.store_nbr}, família {family.name}")
        
        logger.info(f"Total de previsões inseridas: {total_records}")
        return total_records


if __name__ == "__main__":
    """Executar testes de banco de dados."""
    logger.info("Iniciando testes de banco de dados")
    
    try:
        prediction_id = test_add_prediction()
        logger.info(f"Previsão criada com ID: {prediction_id}")
        
        metrics = test_add_metrics()
        logger.info(f"Métricas adicionadas: {metrics}")
        
        importance_df = test_add_feature_importance()
        logger.info(f"Importância de atributos adicionada")
        
        # Adicionar mais testes
        historical_count = test_add_more_historical_sales()
        logger.info(f"Adicionados {historical_count} registros históricos de vendas")
        
        predictions_count = test_add_future_predictions()
        logger.info(f"Adicionadas {predictions_count} previsões futuras")
        
        sales_df = test_query_historical_sales()
        logger.info("Consulta de vendas históricas concluída")
        
        logger.info("Testes concluídos com sucesso")
    
    except Exception as e:
        logger.error(f"Erro durante os testes: {e}")
        sys.exit(1) 