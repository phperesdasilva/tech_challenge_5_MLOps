"""Script de entrada — ponto de execução do experimento.

1. Carrega catálogo e base de clientes.
2. Roda simulação para Baseline e Thompson.
3. Salva resultados em `data/experiments/etapa3/`.
4. Imprime tabela resumo no terminal.

Comando usado:
    python run_etapa3_experiment.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from bandit.catalog import load_catalog
from bandit.policies import BaselineFixedPolicy, ThompsonSamplingPolicy
from bandit.simulator import DEFAULT_BANK_PATH, run_simulation

# Carrega variáveis de ambiente
load_dotenv()

OUT_DIR = Path(os.getenv("OUT_DIR", "data/experiments/etapa3"))
SEED = int(os.getenv("SEED", "42"))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    offers = load_catalog()
    arm_ids = [o["arm_id"] for o in offers]
    df = pd.read_parquet(DEFAULT_BANK_PATH)

    policies = [
        BaselineFixedPolicy(),
        ThompsonSamplingPolicy(arm_ids),
    ]

    summaries = []
    all_history = []

    top_arms_per_policy = {}

    for policy in policies:
        rng = np.random.default_rng(SEED)
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
        arm_df.to_csv(OUT_DIR / f"arm_counts_{policy.name()}.csv", index=False)
        top_arms_per_policy[policy.name()] = arm_df

    pd.DataFrame(summaries).to_csv(OUT_DIR / "metrics_summary.csv", index=False)
    pd.DataFrame(all_history).to_parquet(
        OUT_DIR / "metrics_timeseries.parquet", index=False
    )
    print(pd.DataFrame(summaries))

    print("\n=== Braços mais escolhidos por política ===")
    for policy_name, arm_df in top_arms_per_policy.items():
        print(f"\n[{policy_name}]")
        print(arm_df.to_string(index=False))


if __name__ == "__main__":
    main()
