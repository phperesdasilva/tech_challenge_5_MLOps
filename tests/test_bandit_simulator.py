import numpy as np
import pandas as pd

from bandit.policies import BaselineFixedPolicy
from bandit.simulator import optimal_expected_reward, run_simulation


def test_optimal_expected_reward_returns_zero_for_empty_offers():
    assert optimal_expected_reward([]) == 0.0


def test_optimal_expected_reward_returns_max_reward_for_offers():
    offers = [
        {"prior_conversao_sintetica": 0.2, "valor_recompensa": 10},
        {"prior_conversao_sintetica": 0.5, "valor_recompensa": 10},
    ]

    assert optimal_expected_reward(offers) == 5.0


def test_run_simulation_returns_metrics_tracker_for_simple_case():
    policy = BaselineFixedPolicy()
    df_clients = pd.DataFrame([{"age": 30, "housing": "yes"}])
    offers = [
        {
            "id_braco": "0",
            "valor_recompensa": 5,
            "prior_conversao_sintetica": 1.0,
            "regras_elegibilidade": {},
        }
    ]
    rng = np.random.default_rng(0)

    metrics = run_simulation(policy, df_clients, offers, rng, base_date=None)

    assert metrics.impressions >= 1
    assert metrics.cumulative_reward >= 0
    assert metrics.summary(policy.name())["policy"] == "BaselineFixedPolicy"
