"""Script de entrada — experimento com LinUCB contextual.

Compara todas as quatro políticas lado a lado:
  - BaselineFixedPolicy  (determinístico, sem aprendizado)
  - ThompsonSamplingPolicy (Bayesiano, não-contextual)
  - UCB1Policy             (UCB clássico, não-contextual)
  - LinUCBPolicy           (UCB linear, contextual — usa features do cliente)

Diferença em relação ao run_etapa3_experiment.py:
  1. Constrói um BankContextEncoder ajustado no dataset completo.
  2. Passa o encoder apenas para o LinUCB; as outras políticas rodam igual à Etapa 3.
  3. Salva resultados em data/experiments/linucb/ (separado da Etapa 3).

Comando:
    python src/run_linucb_experiment.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from bandit.catalog import load_catalog
from bandit.features import BankContextEncoder
from bandit.policies import (
    BaselineFixedPolicy,
    LinUCBPolicy,
    ThompsonSamplingPolicy,
    UCB1Policy,
)
from bandit.simulator import DEFAULT_BANK_PATH, run_simulation

load_dotenv()

OUT_DIR = Path(os.getenv("LINUCB_OUT_DIR", "data/experiments/linucb"))
SEED = int(os.getenv("SEED", "42"))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    offers = load_catalog()
    arm_ids = [o["arm_id"] for o in offers]
    df = pd.read_parquet(DEFAULT_BANK_PATH)

    # Encoder ajustado no dataset completo — usado apenas pelo LinUCB
    encoder = BankContextEncoder(df)
    print(f"[LinUCB] dimensão do vetor de contexto: {encoder.dim}")
    print(f"[LinUCB] features: {encoder.feature_names()}\n")

    policies = [
        BaselineFixedPolicy(),
        ThompsonSamplingPolicy(arm_ids),
        UCB1Policy(arm_ids),
        LinUCBPolicy(arm_ids, dim=encoder.dim),
    ]

    summaries = []
    all_history = []

    for policy in policies:
        rng = np.random.default_rng(SEED)

        # Passa encoder apenas para LinUCB; as outras políticas recebem None
        enc = encoder if isinstance(policy, LinUCBPolicy) else None

        metrics = run_simulation(policy, df, offers, rng, encoder=enc)
        summaries.append(metrics.summary(policy.name()))

        for row in metrics.history:
            all_history.append({"policy": policy.name(), **row})

        arm_df = pd.DataFrame(
            [
                {"policy": policy.name(), "arm_id": a, "count": c}
                for a, c in metrics.arm_counts.items()
            ]
        )
        arm_df.to_csv(OUT_DIR / f"arm_counts_{policy.name()}.csv", index=False)

    pd.DataFrame(summaries).to_csv(OUT_DIR / "metrics_summary.csv", index=False)
    pd.DataFrame(all_history).to_parquet(
        OUT_DIR / "metrics_timeseries.parquet", index=False
    )

    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
