"""Script de entrada — ponto de execução do experimento.

1. Carrega catálogo e base de clientes.
2. Roda simulação para Baseline e Thompson.
3. Salva resultados em `data/experiments/etapa3/`.
4. Imprime tabela resumo no terminal.

"""

import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from bandit.catalog import load_catalog
from bandit.policies import BaselineFixedPolicy, ThompsonSamplingPolicy
from bandit.simulator import run_simulation
from bandit.tracking import configure_mlflow, log_dataset, log_policy_metrics

# Carrega variáveis de ambiente

class ThompsonSamplingSimulator:
    def __init__(self):
        load_dotenv()

        self.default_bank_path = Path(
            os.getenv("DEFAULT_BANK_PATH", "data/kaggle/processed/clean_bank.parquet")
        )
        self.out_dir = Path(os.getenv("TS_OUT_DIR", "data/experiments/thompson_sampling"))
        self.seed = int(os.getenv("SEED", "42"))

    def run_thompson_sampling(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        configure_mlflow(os.getenv("MLFLOW_EXPERIMENT_TS", "ThompsonSampling"))
        offers = load_catalog()
        arm_ids = [o["id_braco"] for o in offers]
        df = pd.read_parquet(self.default_bank_path)

        policies = [
            BaselineFixedPolicy(),
            ThompsonSamplingPolicy(arm_ids),
        ]

        summaries = []
        all_history = []

        top_arms_per_policy = {}
        run_ids = {}

        for policy in policies:
            rng = np.random.default_rng(self.seed)
            metrics = run_simulation(policy, df, offers, rng)
            summaries.append(metrics.summary(policy.name()))
            for row in metrics.history:
                all_history.append({"policy": policy.name(), **row})

            arm_df = pd.DataFrame(
                [
                    {"policy": policy.name(), "arm_id": a, "count": c}
                    for a, c in metrics.arm_counts.items()
                ]
            ).sort_values("count", ascending=False)
            arm_df.to_csv(self.out_dir / f"arm_counts_{policy.name()}.csv", index=False)
            top_arms_per_policy[policy.name()] = arm_df

            params = {"seed": self.seed}
            if isinstance(policy, BaselineFixedPolicy):
                params["preferred_arm"] = policy.preferred_arm_id
            elif isinstance(policy, ThompsonSamplingPolicy):
                params["alpha0"] = next(iter(policy.alpha.values()))
                params["beta0"] = next(iter(policy.beta.values()))

            with mlflow.start_run(run_name=policy.name()) as run:
                log_dataset(df, source_path=str(self.default_bank_path), name="clean_bank")
                log_policy_metrics(policy.name(), params, metrics)
                run_ids[policy.name()] = run.info.run_id

        pd.DataFrame(summaries).to_csv(self.out_dir / "metrics_summary.csv", index=False)
        pd.DataFrame(all_history).to_csv(
            self.out_dir / "metrics_timeseries.csv", index=False
        )
        print(pd.DataFrame(summaries))

        print("\n=== Braços mais escolhidos por política ===")
        for policy_name, arm_df in top_arms_per_policy.items():
            print(f"\n[{policy_name}]")
            print(arm_df.to_string(index=False))

        # metrics_summary.csv/metrics_timeseries.parquet só existem a partir daqui
        # (são combinados das duas políticas), então os artifacts são anexados
        # a cada run já finalizado via MlflowClient em vez de mlflow.log_artifacts.
        client = mlflow.MlflowClient()
        for run_id in run_ids.values():
            client.log_artifacts(run_id, str(self.out_dir))
