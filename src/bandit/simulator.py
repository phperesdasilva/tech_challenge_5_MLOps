"""Orquestrador da simulação.

Motor principal que conecta política, ambiente e métricas.

Fluxo para cada cliente:
    cliente → braços elegíveis → política escolhe braço → ambiente simula outcome
    → evento entra na fila de delayed rewards → relógio avança
    → recompensas vencidas são processadas → política atualiza + métricas registram

Pontos importantes:
    1. Fila `pending` — trata recompensas atrasadas; update só ocorre quando
       `conversion_time <= sim_clock`.
    2. Relógio simulado — avança a cada impressão (0–30 dias + horas aleatórias).
    3. Dreno final — ao término, processa todas as recompensas pendentes
       restantes na fila.

Dados:
    Lê clientes de `data/kaggle/processed/clean_bank.parquet`.
"""

import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from .catalog import expected_reward, get_eligible_offers
from .environment import OfferEnvironment
from .metrics import MetricsTracker
from .policies import Policy

# Carrega variáveis de ambiente
load_dotenv()

DEFAULT_BANK_PATH = Path(
    os.getenv("DEFAULT_BANK_PATH", "data/kaggle/processed/clean_bank.parquet")
)
BASE_DATE = datetime.fromisoformat(os.getenv("BASE_DATE", "2026-06-01"))
DELAY_SCALE_DAYS = float(os.getenv("DELAY_SCALE_DAYS", "2.0"))


def optimal_expected_reward(eligible_offers: list[dict]) -> float:
    if not eligible_offers:
        return 0.0
    return max(expected_reward(o) for o in eligible_offers)


def run_simulation(
    policy: Policy,
    df_clients: pd.DataFrame,
    offers: list[dict],
    rng: np.random.Generator,
    base_date: datetime = None,
) -> MetricsTracker:
    if base_date is None:
        base_date = BASE_DATE
    env = OfferEnvironment(rng, delay_scale_days=DELAY_SCALE_DAYS)
    metrics = MetricsTracker()
    offer_by_arm = {o["arm_id"]: o for o in offers}

    # fila: (conversion_time, arm_id, reward, optimal_expected)
    pending: list[tuple] = []
    sim_clock = base_date

    for idx, client in df_clients.iterrows():
        client_dict = client.to_dict()
        client_dict["client_id"] = idx
        eligible = get_eligible_offers(client_dict, offers)
        if not eligible:
            continue

        eligible_ids = [o["arm_id"] for o in eligible]
        arm_id = policy.select_arm(eligible_ids)
        offer = offer_by_arm[arm_id]

        impression_time = sim_clock + timedelta(
            days=int(rng.integers(0, 30)),
            hours=int(rng.integers(0, 23)),
        )
        sim_clock = impression_time

        outcome = env.simulate_outcome(offer, impression_time)
        opt_exp = optimal_expected_reward(eligible)

        pending.append(
            (
                outcome["conversion_time"],
                outcome["arm_id"],
                outcome["reward"],
                outcome["converted"],
                opt_exp,
            )
        )

        # processa recompensas cujo delay já venceu
        pending.sort(key=lambda x: x[0])
        while pending and pending[0][0] <= sim_clock:
            conv_time, arm, reward, converted, opt = pending.pop(0)
            policy.update(arm, converted)
            metrics.record_impression(arm, opt, reward)

    # drenar fila restante (fim da simulação)
    while pending:
        _, arm, reward, converted, opt = pending.pop(0)
        policy.update(arm, converted)
        metrics.record_impression(arm, opt, reward)

    return metrics
