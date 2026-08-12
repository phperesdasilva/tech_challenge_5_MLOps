"""
Engenharia de features para o LinUCB.

Converte um dicionário de cliente em um vetor numérico fixo para ser usado
como contexto pela política LinUCB.

Estrutura do vetor (ordem):
    [age_z, balance_z, housing_bin, loan_bin,
     job_ohe..., marital_ohe..., education_ohe...,
     bias=1.0]

As estatísticas (média, std) e vocabulários são calculados uma única vez
a partir do DataFrame de treino (dataset completo) no __init__.

Colunas de histórico de campanhas anteriores (pdays, previous, poutcome) foram
propositalmente deixadas de fora: no ambiente sintético, a conversão de cada
oferta é um valor fixo por oferta (não depende do contexto do cliente), então
essas colunas não carregavam sinal real — e são justamente as menos disponíveis
para um cliente novo, o que dificultava acionar a rota contextual da API.
"""

import numpy as np
import pandas as pd

NUMERIC_COLS = ["age", "balance"]
BINARY_COLS = ["housing", "loan"]
CATEGORICAL_COLS = ["job", "marital", "education"]


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
        n_numeric = len(NUMERIC_COLS)  # age, balance
        n_binary = len(BINARY_COLS)  # housing, loan
        n_ohe = sum(len(v) for v in self._vocabs.values())
        n_bias = 1
        return n_numeric + n_binary + n_ohe + n_bias

    def feature_names(self) -> list[str]:
        """Nomes de cada posição do vetor (útil para debug e notebook)."""
        names = [f"{c}_z" for c in NUMERIC_COLS]
        names += [f"{c}_bin" for c in BINARY_COLS]
        for col in CATEGORICAL_COLS:
            names += [f"{col}_{cat}" for cat in self._vocabs[col]]
        names.append("bias")
        return names
