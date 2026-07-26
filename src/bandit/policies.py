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

    Modelo linear INDEPENDENTE por braço (por isso "disjoint"): cada oferta
    tem seu próprio "aprendizado" de quais clientes respondem bem a ela.

    Para cada braço `a`, guardamos duas estruturas que resumem tudo que já
    observamos sobre esse braço:
      - contexts_seen_matrix (d×d): acumula os contextos (clientes) já vistos
        por esse braço → começa como matriz identidade (np.eye(d))
      - reward_weighted_context_sum (d,): acumula contexto * recompensa,
        ou seja, "para onde os clientes que converteram apontavam" → começa em zero

    A partir dessas duas estruturas, calculamos:
      - learned_weights = contexts_seen_matrix⁻¹ · reward_weighted_context_sum
        (o vetor de pesos aprendido — o quanto cada feature "pesa" na previsão)
      - score = learned_weights · x + alpha · √(x^T · contexts_seen_matrix⁻¹ · x)
        (previsão de recompensa + bônus de incerteza)

    O segundo termo (bônus de exploração) fica grande quando o braço ainda
    não foi testado em contextos parecidos com o cliente atual — isso empurra
    o algoritmo a experimentar braços incertos, em vez de só repetir o que já
    parece bom.

    Parâmetros
    ----------
    arm_ids : lista de IDs dos braços (igual às outras políticas)
    dim     : dimensão do vetor de contexto (vem de BankContextEncoder.dim)
    alpha   : peso do bônus de exploração — via LINUCB_ALPHA ou padrão 1.0
              (alpha maior = explora mais; alpha menor = confia mais no que já aprendeu)
    """

    def __init__(self, arm_ids: list[str], dim: int, alpha: float = None) -> None:
        if alpha is None:
            alpha = float(os.getenv("LINUCB_ALPHA", "1.0"))
        self.alpha = alpha
        self.dim = dim

        # Uma matriz e um vetor por braço, inicializados "do zero":
        # identidade = ainda não vimos nenhum contexto; zeros = nenhuma recompensa acumulada.
        self.contexts_seen_matrix: dict[str, np.ndarray] = {a: np.eye(dim) for a in arm_ids}
        self.reward_weighted_context_sum: dict[str, np.ndarray] = {a: np.zeros(dim) for a in arm_ids}


    def _ucb_score(self, arm_id: str, client_context: np.ndarray) -> float:
        """Estima o quão bom é oferecer `arm_id` para o cliente descrito por `client_context`.

        score = previsão de recompensa (exploitation) + incerteza (exploration)
        """
        inverse_contexts_matrix = np.linalg.inv(self.contexts_seen_matrix[arm_id])
        learned_weights = inverse_contexts_matrix @ self.reward_weighted_context_sum[arm_id]

        predicted_reward = learned_weights @ client_context  # "achamos que esse cliente vai responder assim"

        # Quanto menos contexto parecido com este já foi visto por este braço,
        # maior a incerteza — e maior o incentivo a explorar.
        uncertainty_bonus = self.alpha * np.sqrt(
            client_context @ inverse_contexts_matrix @ client_context
        )

        return predicted_reward + uncertainty_bonus

    def select_arm(self, eligible_arm_ids: list[str], **kwargs) -> str:
        # O contexto é o vetor numpy gerado pelo BankContextEncoder para este cliente.
        client_context: np.ndarray = kwargs.get("context")
        if client_context is None:
            raise ValueError("LinUCBPolicy exige o kwarg 'context' (np.ndarray) em select_arm().")
        # Escolhe o braço com o maior score entre os elegíveis para este cliente.
        return max(eligible_arm_ids, key=lambda arm_id: self._ucb_score(arm_id, client_context))

    def update(self, arm_id: str, success: bool, **kwargs) -> None:
        # O contexto aqui precisa ser o MESMO vetor usado na chamada de select_arm
        # que gerou esta interação — é assim que a política associa o resultado
        # (sucesso ou não) ao perfil de cliente que o gerou.
        client_context: np.ndarray = kwargs.get("context")
        if client_context is None:
            raise ValueError("LinUCBPolicy exige o kwarg 'context' (np.ndarray) em update().")
        reward = 1.0 if success else 0.0

        # "Vimos mais um cliente com este perfil" — reduz a incerteza nessa direção.
        self.contexts_seen_matrix[arm_id] += np.outer(client_context, client_context)
        # Acumula o sinal de recompensa na direção do contexto observado.
        self.reward_weighted_context_sum[arm_id] += reward * client_context
