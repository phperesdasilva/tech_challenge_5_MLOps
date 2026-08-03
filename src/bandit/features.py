"""
Engenharia de features para o LinUCB.

Converte um dicionário de cliente em um vetor numérico fixo para ser usado
como contexto pela política LinUCB.

Estrutura do vetor (ordem):
    [age_z, balance_z, pdays_z, previous_z, pdays_was_contacted, housing_bin, loan_bin,
     job_ohe..., marital_ohe..., education_ohe..., poutcome_ohe...,
     bias=1.0]

As estatísticas (média, std) e vocabulários são calculados uma única vez
a partir do DataFrame de treino (dataset completo) no __init__.
"""

import numpy as np
import pandas as pd

NUMERIC_COLS = ["age", "balance", "pdays", "previous"]
BINARY_COLS = ["housing", "loan"]
CATEGORICAL_COLS = ["job", "marital", "education", "poutcome"]


class BankContextEncoder:
    """Codifica clientes do Bank Marketing em vetores contextuais para LinUCB."""

    def __init__(self, df_train: pd.DataFrame) -> None:
        self._means: dict[str, float] = {}
        self._stds: dict[str, float] = {}
        self._vocabs: dict[str, list[str]] = {}

        for col in NUMERIC_COLS:
            self._means[col] = float(df_train[col].mean())
            std = float(df_train[col].std())
            self._stds[col] = std if std > 0 else 1.0

        for col in CATEGORICAL_COLS:
            self._vocabs[col] = sorted(df_train[col].dropna().unique().tolist())

    def encode(self, client: dict) -> np.ndarray:
        """Retorna vetor 1D float64 de tamanho `self.dim`."""
        parts: list[float] = []

        # Numéricas z-score
        for col in NUMERIC_COLS:
            val = float(client.get(col, 0.0))
            parts.append((val - self._means[col]) / self._stds[col])

        # Flag: cliente já foi contactado antes (pdays == -1 significa nunca)
        pdays_val = float(client.get("pdays", -1))
        parts.append(1.0 if pdays_val >= 0 else 0.0)

        # Binárias yes/no
        for col in BINARY_COLS:
            parts.append(1.0 if client.get(col) == "yes" else 0.0)

        # One-hot para categóricas
        for col in CATEGORICAL_COLS:
            val = client.get(col)
            for cat in self._vocabs[col]:
                parts.append(1.0 if val == cat else 0.0)

        # Bias
        parts.append(1.0)

        return np.array(parts, dtype=np.float64)

    @property
    def dim(self) -> int:
        """Dimensão do vetor de contexto."""
        n_numeric = len(NUMERIC_COLS)  # age, balance, pdays, previous
        n_pdays_flag = 1
        n_binary = len(BINARY_COLS)  # housing, loan
        n_ohe = sum(len(v) for v in self._vocabs.values())
        n_bias = 1
        return n_numeric + n_pdays_flag + n_binary + n_ohe + n_bias

    def feature_names(self) -> list[str]:
        """Nomes de cada posição do vetor (útil para debug e notebook)."""
        names = [f"{c}_z" for c in NUMERIC_COLS]
        names.append("pdays_was_contacted")
        names += [f"{c}_bin" for c in BINARY_COLS]
        for col in CATEGORICAL_COLS:
            names += [f"{col}_{cat}" for cat in self._vocabs[col]]
        names.append("bias")
        return names
