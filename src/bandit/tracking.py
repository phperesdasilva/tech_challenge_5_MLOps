"""Integração com MLflow para rastrear as execuções dos experimentos.

Este módulo não gerencia o ciclo de vida do run; isso fica a
cargo de cada simulador, já que LinUCB roda uma política por execução e o
Thompson Sampling roda várias políticas no mesmo run.
"""

from bandit.metrics import MetricsTracker
from dotenv import load_dotenv
import mlflow
import os

load_dotenv()


def configure_mlflow(experiment_name: str) -> None:
    """Aponta o MLflow para o tracking store local e seleciona o experimento."""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "mlruns"))
    mlflow.set_experiment(experiment_name)


def log_policy_metrics(policy_name: str, params: dict, metrics: MetricsTracker) -> None:
    """Loga params e métricas de uma política no run MLflow ativo no momento da chamada."""
    mlflow.log_param("policy", policy_name)
    mlflow.log_params(params)

    for row in metrics.history:
        mlflow.log_metrics({k: v for k, v in row.items() if k != "step"}, step=row["step"])

    summary = metrics.summary(policy_name)
    mlflow.log_metrics({k: v for k, v in summary.items() if k != "policy"})
