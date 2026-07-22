import numpy as np

from bandit.policies import BaselineFixedPolicy, ThompsonSamplingPolicy


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
