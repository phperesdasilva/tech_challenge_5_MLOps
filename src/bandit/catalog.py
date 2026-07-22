import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

DEFAULT_CATALOG_PATH = Path(
    os.getenv(
        "DEFAULT_CATALOG_PATH",
        "data/kaggle/synthetic_enrichment/offer_catalog.json",
    )
)


_OFFER_FIELD_MAP = {
    "id_braco": "arm_id",
    "nome_oferta": "offer_name",
    "tipo_oferta": "offer_type",
    "descricao": "description",
    "prior_conversao_sintetica": "synthetic_conversion_prior",
    "valor_recompensa": "reward_value",
    "regras_elegibilidade": "eligibility_rules",
}
_RULE_FIELD_MAP = {
    "idade_minima": "min_age",
    "requer_emprestimo_habitacional": "requires_housing_loan",
}


def _translate_offer(offer: dict) -> dict:
    """Normaliza ofertas do catálogo em português (schema usado pelo pipeline de
    RAG/LLM) para as chaves em inglês esperadas pelo restante do módulo bandit."""
    translated = {_OFFER_FIELD_MAP.get(k, k): v for k, v in offer.items()}
    if "eligibility_rules" in translated:
        translated["eligibility_rules"] = {
            _RULE_FIELD_MAP.get(k, k): v
            for k, v in translated["eligibility_rules"].items()
        }
    return translated


def load_catalog(path: Path = DEFAULT_CATALOG_PATH) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    offers = data.get("offers", data.get("ofertas", []))
    return [_translate_offer(o) for o in offers]


def is_eligible(client: dict, rules: dict) -> bool:
    if client["age"] < rules.get("min_age", 0):
        return False
    if rules.get("requires_housing_loan") and client.get("housing") == "no":
        return False
    return True


def get_eligible_offers(client: dict, offers: list[dict]) -> list[dict]:
    return [o for o in offers if is_eligible(client, o["eligibility_rules"])]


def expected_reward(offer: dict) -> float:
    return offer["synthetic_conversion_prior"] * offer["reward_value"]
