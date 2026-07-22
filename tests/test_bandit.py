import numpy as np
import pytest

from bandit.catalog import expected_reward, get_eligible_offers, is_eligible
from bandit.metrics import MetricsTracker
from bandit.policies import BaselineFixedPolicy, ThompsonSamplingPolicy


class TestBaselineFixedPolicy:
    def test_prefers_configured_arm_when_eligible(self):
        policy = BaselineFixedPolicy(preferred_arm_id="0")
        assert policy.select_arm(["0", "1", "2"]) == "0"

    def test_falls_back_to_first_eligible_when_preferred_missing(self):
        policy = BaselineFixedPolicy(preferred_arm_id="0")
        assert policy.select_arm(["2", "3"]) == "2"

    def test_update_is_noop(self):
        policy = BaselineFixedPolicy()
        policy.update("0", True)  # não deve levantar nem alterar estado


class TestThompsonSamplingPolicy:
    def test_update_moves_posterior_toward_success(self):
        policy = ThompsonSamplingPolicy(["0", "1"], rng=np.random.default_rng(0))
        policy.update("0", True)
        policy.update("0", True)
        policy.update("0", False)
        assert policy.posterior_mean("0") == pytest.approx(3 / 5)

    def test_value_weighting_favors_higher_reward_arm_once_learned(self):
        """Sem ponderação por valor, um bandit puro convergiria para o braço de maior
        taxa de conversão (arm 'cheap'); com arm_values, deve preferir 'valuable' assim
        que os posteriores refletirem as taxas reais simuladas."""
        arm_values = {"cheap": 10.0, "valuable": 300.0}
        policy = ThompsonSamplingPolicy(
            ["cheap", "valuable"], rng=np.random.default_rng(1), arm_values=arm_values
        )
        # Estatísticas fortes o suficiente para dominar o ruído da amostragem.
        policy.alpha["cheap"], policy.beta["cheap"] = 1500.0, 8500.0  # ~15% conversão
        policy.alpha["valuable"], policy.beta["valuable"] = 800.0, 9200.0  # ~8% conversão
        picks = [policy.select_arm(["cheap", "valuable"]) for _ in range(200)]
        assert picks.count("valuable") > picks.count("cheap")

    def test_default_arm_values_preserve_conversion_only_behavior(self):
        policy = ThompsonSamplingPolicy(["a", "b"], rng=np.random.default_rng(2))
        assert policy.arm_values == {"a": 1.0, "b": 1.0}


class TestCatalogEligibility:
    def test_is_eligible_checks_age_and_housing(self):
        rules = {"min_age": 25, "requires_housing_loan": True}
        assert is_eligible({"age": 30, "housing": "yes"}, rules) is True
        assert is_eligible({"age": 20, "housing": "yes"}, rules) is False
        assert is_eligible({"age": 30, "housing": "no"}, rules) is False

    def test_get_eligible_offers_filters_by_rules(self):
        offers = [
            {"arm_id": "0", "eligibility_rules": {"min_age": 18, "requires_housing_loan": False}},
            {"arm_id": "2", "eligibility_rules": {"min_age": 25, "requires_housing_loan": True}},
        ]
        client = {"age": 19, "housing": "no"}
        eligible = get_eligible_offers(client, offers)
        assert [o["arm_id"] for o in eligible] == ["0"]

    def test_expected_reward_multiplies_prior_by_value(self):
        offer = {"synthetic_conversion_prior": 0.08, "reward_value": 300.0}
        assert expected_reward(offer) == pytest.approx(24.0)


class TestMetricsTracker:
    def test_summary_reports_conversion_rate_and_reward(self):
        tracker = MetricsTracker()
        tracker.record_impression("0", optimal_expected=24.0, reward=300.0)
        tracker.record_impression("0", optimal_expected=24.0, reward=0.0)
        summary = tracker.summary("test")
        assert summary["impressions"] == 2
        assert summary["conversions"] == 1
        assert summary["conversion_rate"] == pytest.approx(0.5)
        assert summary["cumulative_reward"] == pytest.approx(300.0)
        assert summary["cumulative_regret"] == pytest.approx(24.0 - 300.0 + 24.0 - 0.0)
