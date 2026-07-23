"""
Define como escolher o braço e como aprender
Policy - Tipo: ABC - Lógica: -  Interface : select_arm() + update
BaselineFixedPolicy - Tipo: Deterministico - Lógica: Sempre arm "0" (conta digital) ;; fallback se nao elegível
ThompsonSamplingPolicy - Tipo: Adaptativo - Lógica: Beta(α, β) por braço; amostra θ e escolhe o maior
UCB1Policy - Tipo: Adaptativo - Lógica: Índice UCB: média + bônus de exploração
LinUCBPolicy - Tipo: Contextual - Lógica: UCB linear; pondera features do cliente por arm  ← NOVO (LinUCB)
"""

import os
from abc import ABC, abstractmethod

import numpy as np
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()


class Policy(ABC):
    # **kwargs adicionado para LinUCB: permite passar context=x sem quebrar
    # as políticas existentes, que simplesmente ignoram o kwarg extra.
    @abstractmethod
    def select_arm(self, eligible_arm_ids: list[str], **kwargs) -> str: ...

    @abstractmethod
    def update(self, arm_id: str, success: bool, **kwargs) -> None: ...  # **kwargs — usado no LinUCB

    def name(self) -> str:
        return self.__class__.__name__


class BaselineFixedPolicy(Policy):
    """Sempre arm_id='0' se elegível; fallback ao primeiro elegível."""

    def __init__(self, preferred_arm_id: str = "0"):
        self.preferred_arm_id = preferred_arm_id

    def select_arm(self, eligible_arm_ids: list[str], **kwargs) -> str:  # **kwargs — compatibilidade LinUCB
        if self.preferred_arm_id in eligible_arm_ids:
            return self.preferred_arm_id
        return eligible_arm_ids[0]

    def update(self, arm_id: str, success: bool, **kwargs) -> None:  # **kwargs — compatibilidade LinUCB
        pass  # determinístico — não aprende


class ThompsonSamplingPolicy(Policy):
    def __init__(self, arm_ids: list[str], alpha0: float = None, beta0: float = None):
        if alpha0 is None:
            alpha0 = float(os.getenv("THOMPSON_ALPHA0", "1.0"))
        if beta0 is None:
            beta0 = float(os.getenv("THOMPSON_BETA0", "1.0"))
        self.alpha = {a: alpha0 for a in arm_ids}
        self.beta = {a: beta0 for a in arm_ids}
        self.rng = np.random.default_rng()

    def select_arm(self, eligible_arm_ids: list[str], **kwargs) -> str:  # **kwargs — compatibilidade LinUCB
        samples = {
            a: self.rng.beta(self.alpha[a], self.beta[a]) for a in eligible_arm_ids
        }
        return max(samples, key=samples.get)

    def update(self, arm_id: str, success: bool, **kwargs) -> None:  # **kwargs — compatibilidade LinUCB
        if success:
            self.alpha[arm_id] += 1
        else:
            self.beta[arm_id] += 1


# =============================================================================
# LinUCBPolicy
# Única política contextual do projeto: usa o vetor de features do cliente
# para personalizar a recomendação por perfil.
# =============================================================================
class LinUCBPolicy(Policy):
    """LinUCB Disjoint (Li et al., 2010).

    Modelo linear independente por arm. A seleção pondera as features do
    cliente usando um bônus de incerteza que diminui conforme o arm é
    explorado em contextos similares.

    Para cada arm `a`:
      - A_a (d×d): acumula produtos externos dos contextos vistos → np.eye(d) no início
      - b_a (d,):  acumula reward * contexto                       → zeros no início
      - θ_a = A_a⁻¹ · b_a  (vetor de pesos aprendido)
      - score = θ_a · x + alpha · √(x^T · A_a⁻¹ · x)

    O segundo termo é o bônus de exploração: grande quando o arm ainda não
    foi explorado em contextos parecidos com o cliente atual.

    Parâmetros
    ----------
    arm_ids : lista de IDs dos braços (igual às outras políticas)
    dim     : dimensão do vetor de contexto (vem de BankContextEncoder.dim)
    alpha   : peso do bônus de exploração — via LINUCB_ALPHA ou padrão 1.0
    """

    def __init__(self, arm_ids: list[str], dim: int, alpha: float = None) -> None:
        if alpha is None:
            alpha = float(os.getenv("LINUCB_ALPHA", "1.0"))  # LINUCB_ALPHA — usado no LinUCB
        self.alpha = alpha
        self.dim = dim
        # Uma matriz A e um vetor b por arm — NOVO (LinUCB)
        self.A: dict[str, np.ndarray] = {a: np.eye(dim) for a in arm_ids}
        self.b: dict[str, np.ndarray] = {a: np.zeros(dim) for a in arm_ids}

    def _ucb_score(self, arm_id: str, x: np.ndarray) -> float:
        """Calcula exploitation + exploration para um arm dado o contexto x."""
        A_inv = np.linalg.inv(self.A[arm_id])
        theta = A_inv @ self.b[arm_id]          # pesos aprendidos
        exploitation = theta @ x                 # recompensa esperada
        exploration = self.alpha * np.sqrt(x @ A_inv @ x)  # bônus de incerteza
        return exploitation + exploration

    def select_arm(self, eligible_arm_ids: list[str], **kwargs) -> str:
        # context é o vetor numpy gerado pelo BankContextEncoder — NOVO (LinUCB)
        x: np.ndarray = kwargs.get("context")
        if x is None:
            raise ValueError(
                "LinUCBPolicy exige o kwarg 'context' (np.ndarray) em select_arm()."
            )
        return max(eligible_arm_ids, key=lambda a: self._ucb_score(a, x))

    def update(self, arm_id: str, success: bool, **kwargs) -> None:
        # context deve ser o mesmo vetor usado na seleção — armazenado na fila pending — NOVO (LinUCB)
        x: np.ndarray = kwargs.get("context")
        if x is None:
            raise ValueError(
                "LinUCBPolicy exige o kwarg 'context' (np.ndarray) em update()."
            )
        reward = 1.0 if success else 0.0
        self.A[arm_id] += np.outer(x, x)   # atualiza confiança na direção x
        self.b[arm_id] += reward * x        # acumula sinal de recompensa
