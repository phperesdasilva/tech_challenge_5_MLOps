"""Script de entrada — ponto de execução do experimento.

1. Carrega catálogo e base de clientes.
2. Roda simulação para Baseline e Thompson.
3. Salva resultados em `data/experiments/etapa3/`.
4. Imprime tabela resumo no terminal.

"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from bandit.catalog import load_catalog
from bandit.policies import BaselineFixedPolicy, ThompsonSamplingPolicy
from bandit.simulator import run_simulation

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

        pd.DataFrame(summaries).to_csv(self.out_dir / "metrics_summary.csv", index=False)
        pd.DataFrame(all_history).to_parquet(
            self.out_dir / "metrics_timeseries.parquet", index=False
        )
        print(pd.DataFrame(summaries))

        print("\n=== Braços mais escolhidos por política ===")
        for policy_name, arm_df in top_arms_per_policy.items():
            print(f"\n[{policy_name}]")
            print(arm_df.to_string(index=False))
