import pandas as pd
import numpy as np
import json
import uuid
from datetime import datetime, timedelta

np.random.seed(42)

OFFER_CATALOG_PATH = r'data/kaggle/synthetic_enrichment/offer_catalog.json'
CLEAN_BANK_PATH = r'data/kaggle/processed/clean_bank.parquet'

def load_data():
    df_bank = pd.read_parquet(CLEAN_BANK_PATH)

    with open(OFFER_CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)
        
    return df_bank, catalog['offers']

def is_eligible(client, rules):
    """Verifica se o cliente atende aos guardrails da oferta."""
    if client['age'] < rules.get('min_age', 0):
        return False
    if rules.get('requires_housing_loan') and client['housing'] == 'no':
        return False
    return True

def simulate_mab_environment():
    df_bank, offers = load_data()
    
    offer_events = []
    delayed_rewards = []
    
    # Data base da simulação
    base_date = datetime(2026, 6, 1)

    print("Iniciando simulação de eventos...")
    for index, client in df_bank.iterrows():
        # 1. Filtrar braços elegíveis (Guardrails)
        eligible_offers = [
            offer for offer in offers 
            if is_eligible(client, offer['eligibility_rules'])
        ]

        client['client_id'] = index
        
        if not eligible_offers:
            continue # Cliente não elegível a nada no catálogo
            
        # 2. Política de Log: Escolher um braço aleatório para exploração inicial
        chosen_offer = np.random.choice(eligible_offers)
        
        # Gerar um ID único para a impressão
        event_id = str(uuid.uuid4())
        impression_time = base_date + timedelta(
            days=np.random.randint(0, 30), 
            hours=np.random.randint(0, 23)
        )
        
        # Registrar a impressão (Contexto + Ação)
        offer_events.append({
            "event_id": event_id,
            "client_id": client['client_id'],
            "timestamp": impression_time,
            "arm_id": chosen_offer['arm_id'],
            "context_age": client['age'],
            "context_balance": client['balance']
        })
        
        # 3. Simular a Recompensa (O cliente converteu?)
        # Usamos o prior da oferta. Em um projeto mais avançado, 
        # você pode somar o prior a um peso baseado no saldo do cliente.
        prob_conversion = chosen_offer['synthetic_conversion_prior']
        is_converted = np.random.binomial(1, prob_conversion) == 1
        
        if is_converted:
            # 4. Simular o Atraso (Delayed Reward)
            # Atraso modelado por uma distribuição exponencial (média de 2 dias)
            delay_days = np.random.exponential(scale=2.0)
            conversion_time = impression_time + timedelta(days=delay_days)
            
            delayed_rewards.append({
                "event_id": event_id,
                "conversion_timestamp": conversion_time,
                "reward_value": chosen_offer['reward_value'],
                "arm_id": chosen_offer['arm_id']
            })
        else:
            # Recompensas negativas/nulas explícitas (opcional, dependendo do design do Bandit)
            delayed_rewards.append({
                "event_id": event_id,
                "conversion_timestamp": None,
                "reward_value": 0.0,
                "arm_id": chosen_offer['arm_id']
            })

    # Converter para DataFrames
    df_events = pd.DataFrame(offer_events)
    df_rewards = pd.DataFrame(delayed_rewards)
    
    # Salvar em Parquet
    df_events.to_parquet("data/kaggle/synthetic_enrichment/offer_events.parquet", index=False)
    df_rewards.to_parquet("data/kaggle/synthetic_enrichment/delayed_rewards.parquet", index=False)

    df_events.to_csv("data/kaggle/synthetic_enrichment/offer_events.csv", index=False)
    df_rewards.to_csv("data/kaggle/synthetic_enrichment/delayed_rewards.csv", index=False)
    
    print(f"Gerados {len(df_events)} eventos de impressão.")
    print(f"Geradas {len(df_rewards[df_rewards['reward_value'] > 0])} conversões com sucesso.")
    print("Arquivos Parquet salvos em data/synthetic_enrichment/")

if __name__ == "__main__":
    # Certifique-se de que as pastas existam antes de rodar
    import os
    os.makedirs("data/synthetic_enrichment", exist_ok=True)
    simulate_mab_environment()