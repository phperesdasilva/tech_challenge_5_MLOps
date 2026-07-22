"""Script de entrada — experimento com LinUCB contextual.

Compara todas as quatro políticas lado a lado:
  - BaselineFixedPolicy  (determinístico, sem aprendizado)
  - ThompsonSamplingPolicy (Bayesiano, não-contextual)
  - UCB1Policy             (UCB clássico, não-contextual)
  - LinUCBPolicy           (UCB linear, contextual — usa features do cliente)

Diferença em relação ao run_etapa3_experiment.py:
  1. Constrói um BankContextEncoder ajustado no dataset completo.
  2. Passa o encoder apenas para o LinUCB;
  3. Salva resultados em data/experiments/linucb/.

Comando:
    python src/run_linucb_experiment.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from bandit.catalog import get_eligible_offers, load_catalog
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


def _linucb_profile_analysis(
    policy: LinUCBPolicy,
    df: pd.DataFrame,
    offers: list[dict],
    encoder: BankContextEncoder,
) -> None:
    """Usa o LinUCB já treinado para inferir o braço recomendado por perfil."""
    records = []
    for _, client in df.iterrows():
        client_dict = client.to_dict()
        eligible = get_eligible_offers(client_dict, offers)
        if not eligible:
            continue
        eligible_ids = [o["id_braco"] for o in eligible]
        x = encoder.encode(client_dict)
        chosen = policy.select_arm(eligible_ids, context=x)
        records.append(
            {
                "arm_id": chosen,
                "job": client_dict.get("job", "?"),
                "education": client_dict.get("education", "?"),
                "poutcome": client_dict.get("poutcome", "?"),
                "age": float(client_dict.get("age", 0)),
            }
        )

    df_rec = pd.DataFrame(records)
    df_rec["age_group"] = pd.cut(
        df_rec["age"],
        bins=[0, 30, 40, 50, 60, 200],
        labels=["<30", "30-40", "40-50", "50-60", "60+"],
    )

    print("\n=== LinUCB — Braço predominante por perfil de cliente ===")
    for col in ("job", "education", "poutcome", "age_group"):
        pivot = df_rec.groupby([col, "arm_id"]).size().unstack(fill_value=0)
        pivot["predominante"] = pivot.idxmax(axis=1)
        pivot["total"] = pivot.drop(columns="predominante").sum(axis=1)
        print(f"\n[Por {col}]")
        print(pivot[["predominante", "total"]].to_string())


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    offers = load_catalog()
    arm_ids = [o["id_braco"] for o in offers]
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
    top_arms_per_policy = {}
    trained_linucb: LinUCBPolicy | None = None

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
        ).sort_values("count", ascending=False)
        arm_df.to_csv(OUT_DIR / f"arm_counts_{policy.name()}.csv", index=False)
        top_arms_per_policy[policy.name()] = arm_df

        if isinstance(policy, LinUCBPolicy):
            trained_linucb = policy

    pd.DataFrame(summaries).to_csv(OUT_DIR / "metrics_summary.csv", index=False)
    pd.DataFrame(all_history).to_parquet(
        OUT_DIR / "metrics_timeseries.parquet", index=False
    )

    print(pd.DataFrame(summaries).to_string(index=False))

    print("\n=== Braços mais escolhidos por política ===")
    for policy_name, arm_df in top_arms_per_policy.items():
        print(f"\n[{policy_name}]")
        print(arm_df.to_string(index=False))

    if trained_linucb is not None:
        _linucb_profile_analysis(trained_linucb, df, offers, encoder)


if __name__ == "__main__":
    main()
