"""
Métricas do experimento
Acumula métricas exigidas na Etapa 3.

- `record_impression()` — após cada observação de recompensa:
    - recompensa acumulada
    - regret = `ótimo esperado − recompensa obtida`
    - taxa de conversão
    - contagem por braço
- `exploration_entropy()` — entropia da distribuição de seleção (exploração).
- `summary()` — resumo final para CSV.

O regret usa o melhor braço **elegível** no momento (`prior × reward`), não só o braço escolhido.
"""
import numpy as np
from collections import Counter

class MetricsTracker:
  def __init__(self):
    self.cumulative_reward = 0.0
    self.cumulative_regret = 0.0
    self.impressions = 0
    self.conversions = 0
    self.arm_counts: Counter = Counter()
    self.history: list[dict] = []

  def record_impression(self, arm_id: str, optimal_expected: float, reward: float):
    self.impressions += 1
    self.arm_counts[arm_id] += 1
    self.cumulative_reward += reward
    regret = optimal_expected - reward
    self.cumulative_regret += regret
    if reward > 0:
      self.conversions += 1
    self.history.append({
      "step": self.impressions,
      "cumulative_reward": self.cumulative_reward,
      "cumulative_regret": self.cumulative_regret,
      "conversion_rate": self.conversions / self.impressions,
    })

  def exploration_entropy(self) -> float:
    total = sum(self.arm_counts.values())
    if total == 0:
      return 0.0
    probs = [c / total for c in self.arm_counts.values()]
    return -sum(p * np.log(p) for p in probs if p > 0)

  def summary(self, policy_name: str) -> dict:
    return {
      "policy": policy_name,
      "impressions": self.impressions,
      "conversions": self.conversions,
      "conversion_rate": self.conversions / max(self.impressions, 1),
      "cumulative_reward": self.cumulative_reward,
      "cumulative_regret": self.cumulative_regret,
      "exploration_entropy": self.exploration_entropy(),
    }