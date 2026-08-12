""""
*Simulação de conversão e delay*
Simula o que acontece depois que uma oferta é mostrada.

- `OfferEnvironment.simulate_outcome()`:
    1. Conversão binária (Bernoulli) com probabilidade = `prior_conversao_sintetica`
       da oferta ajustado por um fator de afinidade com o perfil do cliente
       (ver `_fator_afinidade`) — é essa dependência do contexto que dá ao LinUCB
       algo real para aprender; sem ela, a conversão seria igual para qualquer
       cliente e o contexto não teria nenhum sinal.
    2. Se converte: recompensa = `valor_recompensa`, delay exponencial (média 2 dias).
    3. Se não converte: recompensa 0, observada imediatamente (sem delay).

Representa o **mundo real**: conversão não é instantânea e falhas são conhecidas logo.
"""

import os
from datetime import datetime, timedelta

import numpy as np
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()


def _fator_afinidade(offer: dict, client: dict) -> float:
    """"Verdade" oculta do ambiente sintético: o quanto este cliente combina com esta oferta.

    O LinUCB nunca acessa esta função — ele só vê o vetor de contexto (via
    BankContextEncoder) e tenta aprender esta relação a partir das conversões
    observadas. As regras abaixo usam só features que o encoder também
    enxerga (balance, age, housing), para que exista algo realmente
    aprendível a partir do contexto.
    """
    tipo_oferta = offer.get("tipo_oferta")
    balance = float(client.get("balance", 0))
    age = float(client.get("age", 0))

    if tipo_oferta == "credit":  # cartão premium atrai clientes com mais saldo
        return 1.6 if balance >= 1500 else 0.6
    if tipo_oferta == "loan":  # refinanciamento imobiliário: clientes mais velhos e com mais saldo
        return 1.5 if age >= 40 and balance >= 500 else 0.7
    if tipo_oferta == "investmento":  # CDB: perfil mais conservador, saldo mais alto
        return 1.4 if balance >= 1000 else 0.8
    return 1.0  # baseline: sem ajuste, serve de controle


class OfferEnvironment:
    def __init__(self, rng: np.random.Generator, delay_scale_days: float = None):
        if delay_scale_days is None:
            delay_scale_days = float(os.getenv("DELAY_SCALE_DAYS", "2.0"))
        self.rng = rng
        self.delay_scale_days = delay_scale_days

    def simulate_outcome(self, offer: dict, client: dict, impression_time: datetime) -> dict:
        prob_conversao = offer["prior_conversao_sintetica"] * _fator_afinidade(offer, client)
        prob_conversao = min(1.0, max(0.0, prob_conversao))

        converted = self.rng.binomial(1, prob_conversao) == 1
        if converted:
            delay = self.rng.exponential(scale=self.delay_scale_days)
            conversion_time = impression_time + timedelta(days=delay)
            reward = offer["valor_recompensa"]
        else:
            conversion_time = impression_time  # falha observada imediatamente
            reward = 0.0
        return {
            "converted": converted,
            "reward": reward,
            "conversion_time": conversion_time,
            "arm_id": offer["id_braco"],
        }
