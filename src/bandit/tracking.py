"""Integração com MLflow para rastrear as execuções dos experimentos.

Este módulo não gerencia o ciclo de vida do run; isso fica a
cargo de cada simulador, já que LinUCB roda uma política por execução e o
Thompson Sampling roda várias políticas no mesmo run.
"""

import time
import warnings

from bandit.metrics import MetricsTracker
from dotenv import load_dotenv
from mlflow.entities import Metric
from mlflow.utils.validation import MAX_METRICS_PER_BATCH
import mlflow
import mlflow.data
from mlflow.data.sources import LocalArtifactDatasetSource
import os
import pandas as pd

load_dotenv()


def configure_mlflow(experiment_name: str) -> None:
    """Aponta o MLflow para o tracking store local e seleciona o experimento."""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "mlruns"))
    mlflow.set_experiment(experiment_name)


def log_dataset(df: pd.DataFrame, source_path: str, name: str = None, context: str = "training") -> None:
    """Associa o dataset usado ao run ativo (preenche a coluna 'Dataset' no MLflow UI).

    Constrói o DatasetSource explicitamente em vez de passar `source_path` como
    string: o registry interno do MLflow associa arquivos locais a duas entradas
    equivalentes (esquemas "" e "file"), então resolver a partir de uma string
    sempre "bate" com as duas e gera o aviso de ambiguidade
    ("dataset source can be interpreted in multiple ways"). Passar o objeto já
    resolvido evita essa etapa de resolução.
    """
    source = LocalArtifactDatasetSource(uri=source_path)
    dataset = mlflow.data.from_pandas(df, source=source, name=name or os.path.basename(source_path))

    with warnings.catch_warnings():
        # mlflow infere o schema do dataset para exibi-lo no UI e avisa quando
        # há colunas inteiras, pois um modelo *servido* com essas colunas
        # ausentes em produção as receberia como float. Aqui o dataset só é
        # usado para registrar linhagem (qual CSV/parquet gerou o run), não
        # para servir um modelo, então o aviso não se aplica.
        warnings.filterwarnings(
            "ignore",
            message="Hint: Inferred schema contains integer column",
            category=UserWarning,
        )
        mlflow.log_input(dataset, context=context)


def log_policy_metrics(policy_name: str, params: dict, metrics: MetricsTracker) -> None:
    """Loga params e métricas de uma política no run MLflow ativo no momento da chamada.

    `metrics.history` tem uma linha por impressão — para datasets grandes
    (dezenas de milhares de linhas), logar uma métrica por vez via
    `mlflow.log_metrics` faz uma transação por chamada e fica lento. Em vez
    disso, monta todos os pontos e envia em lotes via `MlflowClient.log_batch`.
    """
    mlflow.log_param("policy", policy_name)
    mlflow.log_params(params)

    run_id = mlflow.active_run().info.run_id
    timestamp_ms = int(time.time() * 1000)
    points = [
        Metric(key=key, value=value, timestamp=timestamp_ms, step=row["step"])
        for row in metrics.history
        for key, value in row.items()
        if key != "step"
    ]

    client = mlflow.MlflowClient()
    for i in range(0, len(points), MAX_METRICS_PER_BATCH):
        client.log_batch(run_id, metrics=points[i : i + MAX_METRICS_PER_BATCH])

    # cumulative_reward/cumulative_regret/conversion_rate já estão no histórico por
    # step (com o mesmo valor final) — relogá-los aqui sem step criaria um ponto
    # duplicado em step=0, poluindo a curva no MLflow UI. Loga só o que é exclusivo do summary.
    history_keys = set(metrics.history[-1].keys()) - {"step"} if metrics.history else set()
    summary = metrics.summary(policy_name)
    mlflow.log_metrics(
        {k: v for k, v in summary.items() if k != "policy" and k not in history_keys}
    )
