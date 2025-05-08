#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime, timedelta
import random
from src.database.database import db_session
from src.database.models import Store, ProductFamily, HistoricalSales
from src.database.repository import StoreRepository, ProductFamilyRepository, HistoricalSalesRepository

def main():
    """Adiciona dados de exemplo para as combinações de loja/família principais."""
    print("Adicionando dados de exemplo para combinações loja/família...")
    
    # Combinações que queremos garantir que existem
    target_combinations = [
        (1, "PRODUCE"),
        (1, "FROZEN FOODS"),
        (1, "GROCERY II"),  # Essa já existe, mas vamos verificar
        (1, "LIQUOR,WINE,BEER"),
        (1, "HOME APPLIANCES")
    ]
    
    with db_session() as db:
        # Para cada combinação
        for store_nbr, family_name in target_combinations:
            # Verificar se store e family existem
            store = StoreRepository.get_by_store_nbr(db, store_nbr)
            if not store:
                print(f"Erro: Store {store_nbr} não existe! Pulando...")
                continue
                
            family = ProductFamilyRepository.get_by_name(db, family_name)
            if not family:
                print(f"Erro: Family {family_name} não existe! Pulando...")
                continue
            
            # Verificar se já existem registros
            count = db.query(HistoricalSales).filter(
                HistoricalSales.store_id == store.id,
                HistoricalSales.family_id == family.id
            ).count()
            
            if count > 0:
                print(f"Store {store_nbr}, Family {family_name}: Já existem {count} registros. Pulando...")
                continue
            
            # Gerar dados de exemplo (90 dias)
            sales_data_list = []
            end_date = datetime.now().date()
            
            # Fatores para criar padrões sazonais realistas
            day_of_week_factors = {0: 0.8, 1: 0.9, 2: 1.0, 3: 1.0, 4: 1.2, 5: 1.5, 6: 0.7}
            base_sales = 100 + (hash(family_name) % 100)  # Valor base diferente para cada família
            
            # Gerar 90 dias de dados
            for day_offset in range(90, 0, -1):
                # Calcular data
                current_date = end_date - timedelta(days=day_offset)
                
                # Aplicar fatores sazonais
                weekday = current_date.weekday()
                day_factor = day_of_week_factors.get(weekday, 1.0)
                
                # Calcular vendas com um pouco de aleatoriedade
                sales = base_sales * day_factor * (1 + random.uniform(-0.2, 0.2))
                
                # Determinar se está em promoção (20% de chance)
                onpromotion = random.random() < 0.2
                
                # Se estiver em promoção, aumentar as vendas
                if onpromotion:
                    sales *= 1.3
                
                # Criar registro
                sales_data_list.append({
                    "store_id": store.id,
                    "family_id": family.id,
                    "date": current_date,
                    "sales": sales,
                    "onpromotion": onpromotion
                })
            
            # Inserir no banco de dados
            HistoricalSalesRepository.create_many(db, sales_data_list)
            print(f"Adicionados {len(sales_data_list)} registros para Store {store_nbr}, Family {family_name}")
    
    print("\nOperação concluída!")

if __name__ == "__main__":
    main() 