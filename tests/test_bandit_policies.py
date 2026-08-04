import numpy as np
import pytest

from bandit.policies import BaselineFixedPolicy, LinUCBPolicy, ThompsonSamplingPolicy


def test_baseline_policy_prefers_preferred_arm_when_eligible():
    policy = BaselineFixedPolicy(preferred_arm_id="1")

    selected = policy.select_arm(["0", "1"])

    assert selected == "1"


def test_baseline_policy_falls_back_to_first_eligible_arm():
    policy = BaselineFixedPolicy(preferred_arm_id="9")

    selected = policy.select_arm(["0", "1"])

    assert selected == "0"


def test_baseline_policy_update_is_a_noop():
    policy = BaselineFixedPolicy()

    policy.update("0", True)

    assert policy.preferred_arm_id == "0"


def test_thompson_policy_selects_one_of_eligible_arms():
    policy = ThompsonSamplingPolicy(["0", "1"], alpha0=1.0, beta0=1.0)
    policy.rng = np.random.default_rng(0)

    selected = policy.select_arm(["0", "1"])

    assert selected in {"0", "1"}


def test_thompson_policy_updates_alpha_and_beta_after_feedback():
    policy = ThompsonSamplingPolicy(["0", "1"], alpha0=1.0, beta0=1.0)

    policy.update("0", True)
    policy.update("1", False)

    assert policy.alpha["0"] == 2.0
    assert policy.beta["1"] == 2.0


def test_linucb_initializes_identity_matrices_and_zero_reward_vectors():
    policy = LinUCBPolicy(["0", "1"], dim=3)

    assert np.array_equal(policy.contexts_seen_matrix["0"], np.eye(3))
    assert np.array_equal(policy.contexts_seen_matrix["1"], np.eye(3))
    assert np.array_equal(policy.reward_weighted_context_sum["0"], np.zeros(3))
    assert np.array_equal(policy.reward_weighted_context_sum["1"], np.zeros(3))


def test_linucb_select_arm_requires_context_kwarg():
    policy = LinUCBPolicy(["0", "1"], dim=2)

    with pytest.raises(ValueError):
        policy.select_arm(["0", "1"])


def test_linucb_update_requires_context_kwarg():
    policy = LinUCBPolicy(["0", "1"], dim=2)

    with pytest.raises(ValueError):
        policy.update("0", True)


def test_linucb_selects_first_eligible_arm_when_scores_tie_before_any_update():
    policy = LinUCBPolicy(["0", "1", "2"], dim=2)
    context = np.array([1.0, 0.5])

    selected = policy.select_arm(["0", "1", "2"], context=context)

    assert selected == "0"


def test_linucb_update_accumulates_context_outer_product_and_weighted_reward():
    policy = LinUCBPolicy(["0"], dim=2)
    context = np.array([2.0, 1.0])

    policy.update("0", True, context=context, reward=3.0)

    expected_matrix = np.eye(2) + np.outer(context, context)
    expected_vector = 3.0 * context
    assert np.allclose(policy.contexts_seen_matrix["0"], expected_matrix)
    assert np.allclose(policy.reward_weighted_context_sum["0"], expected_vector)


def test_linucb_update_defaults_reward_to_one_on_success_without_reward_kwarg():
    policy = LinUCBPolicy(["0"], dim=1)

    policy.update("0", True, context=np.array([1.0]))

    assert policy.reward_weighted_context_sum["0"][0] == 1.0


def test_linucb_update_defaults_reward_to_zero_on_failure_without_reward_kwarg():
    policy = LinUCBPolicy(["0"], dim=1)

    policy.update("0", False, context=np.array([1.0]))

    assert policy.reward_weighted_context_sum["0"][0] == 0.0


def test_linucb_prefers_arm_with_learned_positive_reward_for_matching_context():
    policy = LinUCBPolicy(["0", "1"], dim=2, alpha=0.0)
    context = np.array([1.0, 0.0])

    for _ in range(5):
        policy.update("1", True, context=context, reward=1.0)

    selected = policy.select_arm(["0", "1"], context=context)

    assert selected == "1"
