import numpy as np
import pandas as pd

from bandit.policies import BaselineFixedPolicy
from bandit.simulator import optimal_expected_reward, run_simulation


def test_optimal_expected_reward_returns_zero_for_empty_offers():
    assert optimal_expected_reward([]) == 0.0


def test_optimal_expected_reward_returns_max_reward_for_offers():
    offers = [
        {"synthetic_conversion_prior": 0.2, "reward_value": 10},
        {"synthetic_conversion_prior": 0.5, "reward_value": 10},
    ]

    assert optimal_expected_reward(offers) == 5.0


def test_run_simulation_returns_metrics_tracker_for_simple_case():
    policy = BaselineFixedPolicy()
    df_clients = pd.DataFrame([{"age": 30, "housing": "yes"}])
    offers = [{"arm_id": "0", "reward_value": 5, "synthetic_conversion_prior": 1.0, "eligibility_rules": {}}]
    rng = np.random.default_rng(0)

    metrics = run_simulation(policy, df_clients, offers, rng, base_date=None)

    assert metrics.impressions >= 1
    assert metrics.cumulative_reward >= 0
    assert metrics.summary(policy.name())["policy"] == "BaselineFixedPolicy"
