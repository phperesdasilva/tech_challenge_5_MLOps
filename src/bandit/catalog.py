import json
from pathlib import Path
from typing import Any

DEFAULT_CATALOG_PATH = Path("data/kaggle/synthetic_enrichment/offer_catalog.json")

def load_catalog(path: Path = DEFAULT_CATALOG_PATH) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["offers"]

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