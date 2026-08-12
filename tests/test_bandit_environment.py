from datetime import datetime

import numpy as np

from bandit.environment import OfferEnvironment, _fator_afinidade


def test_simulate_outcome_marks_conversion_and_sets_reward():
    rng = np.random.default_rng(123)
    env = OfferEnvironment(rng, delay_scale_days=1.0)
    offer = {"id_braco": "1", "prior_conversao_sintetica": 1.0, "valor_recompensa": 10}

    outcome = env.simulate_outcome(offer, {}, datetime(2026, 1, 1, 12, 0))

    assert outcome["converted"] is True
    assert outcome["reward"] == 10
    assert outcome["arm_id"] == "1"
    assert outcome["conversion_time"] > datetime(2026, 1, 1, 12, 0)


def test_simulate_outcome_handles_non_conversion_without_delay():
    rng = np.random.default_rng(7)
    env = OfferEnvironment(rng, delay_scale_days=0.5)
    offer = {"id_braco": "2", "prior_conversao_sintetica": 0.0, "valor_recompensa": 5}

    outcome = env.simulate_outcome(offer, {}, datetime(2026, 1, 1, 12, 0))

    assert outcome["converted"] is False
    assert outcome["reward"] == 0.0
    assert outcome["conversion_time"] == datetime(2026, 1, 1, 12, 0)


def test_simulate_outcome_clamps_probability_when_affinity_pushes_prior_above_one():
    rng = np.random.default_rng(0)
    env = OfferEnvironment(rng, delay_scale_days=1.0)
    offer = {"id_braco": "1", "tipo_oferta": "credit", "prior_conversao_sintetica": 0.9, "valor_recompensa": 10}
    client = {"balance": 2000, "age": 30}  # fator 1.6 -> 0.9*1.6=1.44, precisa ser limitado a 1.0

    outcome = env.simulate_outcome(offer, client, datetime(2026, 1, 1, 12, 0))

    assert outcome["converted"] is True  # prob=1.0 -> conversão garantida


def test_fator_afinidade_favors_higher_balance_for_credit_offer():
    offer = {"tipo_oferta": "credit"}

    assert _fator_afinidade(offer, {"balance": 2000, "age": 30}) == 1.6
    assert _fator_afinidade(offer, {"balance": 500, "age": 30}) == 0.6


def test_fator_afinidade_favors_older_wealthier_clients_for_loan_offer():
    offer = {"tipo_oferta": "loan"}

    assert _fator_afinidade(offer, {"balance": 1000, "age": 45}) == 1.5
    assert _fator_afinidade(offer, {"balance": 1000, "age": 25}) == 0.7
    assert _fator_afinidade(offer, {"balance": 100, "age": 45}) == 0.7


def test_fator_afinidade_favors_higher_balance_for_investment_offer():
    offer = {"tipo_oferta": "investmento"}

    assert _fator_afinidade(offer, {"balance": 1500, "age": 30}) == 1.4
    assert _fator_afinidade(offer, {"balance": 200, "age": 30}) == 0.8


def test_fator_afinidade_is_neutral_for_baseline_offer():
    offer = {"tipo_oferta": "baseline"}

    assert _fator_afinidade(offer, {"balance": 1_000_000, "age": 90}) == 1.0
