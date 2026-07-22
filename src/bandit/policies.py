"""
Define como escolher o braço e como aprender
Policy - Tipo: ABC - Lógica: -  Interface : select_arm() + update
BaselineFixedPolicy - Tipo: Deterministico - Lógica: Sempre arm "0" (conta digital) ;; fallback se nao elegível
ThompsonSamplingPolicy - Tipo: Adaptativo - Lógica: Beta(α, β) por braço; amostra θ e escolhe o maior
"""

import os
from abc import ABC, abstractmethod

import numpy as np
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()


class Policy(ABC):
    @abstractmethod
    def select_arm(self, eligible_arm_ids: list[str]) -> str: ...

    @abstractmethod
    def update(self, arm_id: str, success: bool) -> None: ...

    def name(self) -> str:
        return self.__class__.__name__


class BaselineFixedPolicy(Policy):
    """Sempre arm_id='0' se elegível; fallback ao primeiro elegível."""

    def __init__(self, preferred_arm_id: str = "0"):
        self.preferred_arm_id = preferred_arm_id

    def select_arm(self, eligible_arm_ids: list[str]) -> str:
        if self.preferred_arm_id in eligible_arm_ids:
            return self.preferred_arm_id
        return eligible_arm_ids[0]

    def update(self, arm_id: str, success: bool) -> None:
        pass  # determinístico — não aprende


class ThompsonSamplingPolicy(Policy):
    """Beta(α, β) por braço, com amostra ponderada pelo valor de negócio do braço.

    `arm_values` pondera a amostra θ pelo valor de recompensa de cada braço
    (θ × valor), evitando que a política convirja para o braço de maior taxa de
    conversão quando ele não é o de maior valor esperado. Sem `arm_values`, o
    comportamento é o bandit puro por taxa de conversão (todos os pesos = 1.0).
    """

    def __init__(
        self,
        arm_ids: list[str],
        alpha0: float = None,
        beta0: float = None,
        rng: np.random.Generator = None,
        arm_values: dict[str, float] = None,
    ):
        if alpha0 is None:
            alpha0 = float(os.getenv("THOMPSON_ALPHA0", "1.0"))
        if beta0 is None:
            beta0 = float(os.getenv("THOMPSON_BETA0", "1.0"))
        self.alpha = {a: alpha0 for a in arm_ids}
        self.beta = {a: beta0 for a in arm_ids}
        self.rng = rng if rng is not None else np.random.default_rng()
        self.arm_values = arm_values if arm_values is not None else {a: 1.0 for a in arm_ids}

    def select_arm(self, eligible_arm_ids: list[str]) -> str:
        samples = {
            a: self.rng.beta(self.alpha[a], self.beta[a]) * self.arm_values.get(a, 1.0)
            for a in eligible_arm_ids
        }
        return max(samples, key=samples.get)

    def update(self, arm_id: str, success: bool) -> None:
        if success:
            self.alpha[arm_id] += 1
        else:
            self.beta[arm_id] += 1

    def posterior_mean(self, arm_id: str) -> float:
        return self.alpha[arm_id] / (self.alpha[arm_id] + self.beta[arm_id])

    def posterior_expected_value(self, arm_id: str) -> float:
        return self.posterior_mean(arm_id) * self.arm_values.get(arm_id, 1.0)


