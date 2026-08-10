import mlflow
import pandas as pd

from bandit.metrics import MetricsTracker
from bandit.tracking import configure_mlflow, log_dataset, log_policy_metrics


def _configure_isolated_mlflow(tmp_path, monkeypatch, experiment_name: str) -> None:
    """Aponta o MLflow para um SQLite temporário, isolado dos runs reais em mlflow.db."""
    db_path = tmp_path / "mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{db_path}")
    configure_mlflow(experiment_name)


def test_configure_mlflow_creates_experiment(tmp_path, monkeypatch):
    _configure_isolated_mlflow(tmp_path, monkeypatch, "TestExperiment")

    experiment = mlflow.get_experiment_by_name("TestExperiment")

    assert experiment is not None
    assert experiment.lifecycle_stage == "active"


def test_log_policy_metrics_records_params_and_metrics(tmp_path, monkeypatch):
    _configure_isolated_mlflow(tmp_path, monkeypatch, "TestPolicyMetrics")

    tracker = MetricsTracker()
    tracker.record_impression("0", optimal_expected=1.0, reward=1.0)
    tracker.record_impression("1", optimal_expected=1.0, reward=0.0)

    with mlflow.start_run(run_name="policy_run") as run:
        log_policy_metrics("TestPolicy", {"seed": 42, "alpha": 1.0}, tracker)
        run_id = run.info.run_id

    logged_run = mlflow.MlflowClient().get_run(run_id)

    assert logged_run.data.params["policy"] == "TestPolicy"
    assert logged_run.data.params["seed"] == "42"
    assert logged_run.data.params["alpha"] == "1.0"

    summary = tracker.summary("TestPolicy")
    assert logged_run.data.metrics["cumulative_reward"] == summary["cumulative_reward"]
    assert logged_run.data.metrics["conversion_rate"] == summary["conversion_rate"]
    assert logged_run.data.metrics["impressions"] == summary["impressions"]

    history = mlflow.MlflowClient().get_metric_history(run_id, "cumulative_reward")
    assert len(history) == len(tracker.history)


def test_log_dataset_attaches_dataset_input(tmp_path, monkeypatch):
    _configure_isolated_mlflow(tmp_path, monkeypatch, "TestDataset")

    df = pd.DataFrame({"age": [30, 40, 50]})

    with mlflow.start_run(run_name="dataset_run") as run:
        log_dataset(df, source_path="data/kaggle/processed", name="clean_bank")
        run_id = run.info.run_id

    logged_run = mlflow.MlflowClient().get_run(run_id)

    assert len(logged_run.inputs.dataset_inputs) == 1
    assert logged_run.inputs.dataset_inputs[0].dataset.name == "clean_bank"
