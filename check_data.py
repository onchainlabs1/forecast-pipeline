#!/usr/bin/env python
# -*- coding: utf-8 -*-

from src.database.database import db_session
from src.database.models import Store, ProductFamily, HistoricalSales

def main():
    """Verifica combinações de loja/família existentes no banco de dados."""
    print("Verificando combinações de loja/família existentes...")
    
    with db_session() as db:
        # Buscar todas as combinações existentes
        result = db.query(Store.store_nbr, ProductFamily.name)\
            .join(HistoricalSales, HistoricalSales.store_id == Store.id)\
            .join(ProductFamily, HistoricalSales.family_id == ProductFamily.id)\
            .distinct()\
            .all()
        
        if not result:
            print("Nenhuma combinação encontrada!")
            return
            
        print(f"Encontradas {len(result)} combinações loja/família:")
        for r in result:
            print(f"Store: {r[0]}, Family: {r[1]}")
            
        # Verificar algumas combinações específicas
        combos_to_check = [
            (1, "PRODUCE"),
            (1, "FROZEN FOODS"),
            (1, "GROCERY II"),
            (1, "LIQUOR,WINE,BEER"),
            (1, "HOME APPLIANCES")
        ]
        
        print("\nVerificando combinações específicas:")
        for store_nbr, family_name in combos_to_check:
            # Buscar store e family
            store = db.query(Store).filter(Store.store_nbr == store_nbr).first()
            family = db.query(ProductFamily).filter(ProductFamily.name == family_name).first()
            
            if not store or not family:
                print(f"Store {store_nbr} ou Family {family_name} não existe")
                continue
                
            # Verificar se existem registros
            count = db.query(HistoricalSales)\
                .filter(HistoricalSales.store_id == store.id, 
                        HistoricalSales.family_id == family.id)\
                .count()
                
            if count > 0:
                print(f"✅ Store {store_nbr}, Family {family_name}: {count} registros")
            else:
                print(f"❌ Store {store_nbr}, Family {family_name}: nenhum registro")

if __name__ == "__main__":
    main() 