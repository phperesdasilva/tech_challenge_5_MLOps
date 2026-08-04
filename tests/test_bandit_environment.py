from datetime import datetime

import numpy as np

from bandit.environment import OfferEnvironment


def test_simulate_outcome_marks_conversion_and_sets_reward():
    rng = np.random.default_rng(123)
    env = OfferEnvironment(rng, delay_scale_days=1.0)
    offer = {"id_braco": "1", "prior_conversao_sintetica": 1.0, "valor_recompensa": 10}

    outcome = env.simulate_outcome(offer, datetime(2026, 1, 1, 12, 0))

    assert outcome["converted"] is True
    assert outcome["reward"] == 10
    assert outcome["arm_id"] == "1"
    assert outcome["conversion_time"] > datetime(2026, 1, 1, 12, 0)


def test_simulate_outcome_handles_non_conversion_without_delay():
    rng = np.random.default_rng(7)
    env = OfferEnvironment(rng, delay_scale_days=0.5)
    offer = {"id_braco": "2", "prior_conversao_sintetica": 0.0, "valor_recompensa": 5}

    outcome = env.simulate_outcome(offer, datetime(2026, 1, 1, 12, 0))

    assert outcome["converted"] is False
    assert outcome["reward"] == 0.0
    assert outcome["conversion_time"] == datetime(2026, 1, 1, 12, 0)
