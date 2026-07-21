from bandit.metrics import MetricsTracker


def test_record_impression_updates_reward_and_regret():
    tracker = MetricsTracker()

    tracker.record_impression("0", optimal_expected=5.0, reward=3.0)

    assert tracker.impressions == 1
    assert tracker.cumulative_reward == 3.0
    assert tracker.cumulative_regret == 2.0
    assert tracker.arm_counts["0"] == 1
    assert tracker.history[0]["step"] == 1


def test_exploration_entropy_and_summary_are_computed_correctly():
    tracker = MetricsTracker()
    tracker.record_impression("0", optimal_expected=4.0, reward=2.0)
    tracker.record_impression("1", optimal_expected=4.0, reward=0.0)

    entropy = tracker.exploration_entropy()
    summary = tracker.summary("Baseline")

    assert entropy > 0
    assert summary["policy"] == "Baseline"
    assert summary["impressions"] == 2
    assert summary["conversions"] == 1
    assert summary["conversion_rate"] == 0.5
