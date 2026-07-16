"""Guardrails do assistente.

Camada inicial de proteção (será expandida no system-card da Etapa 8). Cobre:
- Atributos proibidos para decisão (renda, gênero, raça, patrimônio, etc.).
- Detecção básica de tentativa de prompt-injection.
- Nota de "humano no loop" para ofertas sensíveis (crédito/refinanciamento).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Atributos que o enunciado proíbe usar como critério de decisão
PROTECTED_ATTRIBUTES = [
    "renda",
    "salário",
    "salario",
    "gênero",
    "genero",
    "sexo",
    "raça",
    "raca",
    "cor da pele",
    "etnia",
    "religião",
    "religiao",
    "orientação sexual",
    "orientacao sexual",
    "patrimônio",
    "patrimonio",
]

# Padrões simples de injeção de prompt
INJECTION_PATTERNS = [
    r"ignore (as |suas |todas as )?instru",
    r"ignore (the |all )?previous",
    r"desconsidere (as |suas )?instru",
    r"disregard (the |all )?",
    r"revele? (o |seu )?(system prompt|prompt de sistema)",
    r"aja como .* sem restri",
    r"jailbreak",
]

# Ofertas que exigem revisão humana antes da efetivação
SENSITIVE_OFFER_TYPES = {"credit", "loan"}


@dataclass
class GuardResult:
    allowed: bool
    reason: str = ""


def check_input(text: str) -> GuardResult:
    """Valida a entrada do usuário antes de chamar o LLM."""
    low = text.lower()

    for attr in PROTECTED_ATTRIBUTES:
        # bloqueia apenas quando o atributo é usado como CRITÉRIO de decisão
        if attr in low and re.search(
            r"(decid|escolh|ofert|recomend|seleci|priori|segment).{0,40}" + re.escape(attr)
            + r"|" + re.escape(attr) + r".{0,40}(decid|escolh|ofert|recomend|seleci|priori|segment)",
            low,
        ):
            return GuardResult(
                allowed=False,
                reason=(
                    f"Solicitação bloqueada: o atributo protegido '{attr}' não pode ser usado "
                    "como critério de decisão ou recomendação (restrição do desafio/LGPD)."
                ),
            )

    for pat in INJECTION_PATTERNS:
        if re.search(pat, low):
            return GuardResult(
                allowed=False,
                reason="Solicitação bloqueada: possível tentativa de manipulação das instruções do assistente.",
            )

    return GuardResult(allowed=True)


def human_in_the_loop_note(offer_type: str) -> str:
    """Retorna a nota de revisão humana quando a oferta é sensível."""
    if (offer_type or "").lower() in SENSITIVE_OFFER_TYPES:
        return (
            "\n\n⚠️ Decisão sensível: esta é uma sugestão automatizada e deve passar por "
            "revisão humana antes da efetivação (humano no loop)."
        )
    return ""


def scrub_output(text: str) -> str:
    """Pós-processa a saída. Por ora, apenas garante que não há vazamento explícito
    de instrução de sistema. Hooks adicionais entram no system-card da Etapa 8."""
    return text.replace("system prompt", "[redigido]")
