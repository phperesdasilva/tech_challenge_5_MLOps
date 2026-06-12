""""
*Simulação de conversão e delay*
Simula o que acontece depois que uma oferta é mostrada.

- `OfferEnvironment.simulate_outcome()`:
    1. Conversão binária com `synthetic_conversion_prior` (Bernoulli).
    2. Se converte: recompensa = `reward_value`, delay exponencial (média 2 dias).
    3. Se não converte: recompensa 0, observada imediatamente (sem delay).

Representa o **mundo real**: conversão não é instantânea e falhas são conhecidas logo.
"""

import numpy as np
from datetime import datetime, timedelta

class OfferEnvironment:
  def __init__(self, rng: np.random.Generator, delay_scale_days: float = 2.0):
    self.rng = rng
    self.delay_scale_days = delay_scale_days

  def simulate_outcome(self, offer: dict, impression_time: datetime) -> dict:
    converted = self.rng.binomial(1, offer["synthetic_conversion_prior"]) == 1
    if converted:
      delay = self.rng.exponential(scale=self.delay_scale_days)
      conversion_time = impression_time + timedelta(days=delay)
      reward = offer["reward_value"]
    else:
      conversion_time = impression_time  # falha observada imediatamente
      reward = 0.0
    return {
      "converted": converted,
      "reward": reward,
      "conversion_time": conversion_time,
      "arm_id": offer["arm_id"],
    }