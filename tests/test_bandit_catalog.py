import json

from bandit.catalog import expected_reward, get_eligible_offers, is_eligible, load_catalog


def test_load_catalog_reads_offers_from_json(tmp_path):
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps({"offers": [{"id": "A"}]}), encoding="utf-8")

    assert load_catalog(catalog_path) == [{"id": "A"}]


def test_load_catalog_returns_list_of_offers_when_file_exists(tmp_path):
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps({"offers": [{"id": "B"}, {"id": "C"}]}), encoding="utf-8")

    offers = load_catalog(catalog_path)

    assert len(offers) == 2
    assert offers[0]["id"] == "B"


def test_is_eligible_respects_min_age_and_housing_rule():
    client = {"age": 30, "housing": "no"}
    rules = {"min_age": 18, "requires_housing_loan": True}

    assert is_eligible(client, rules) is False


def test_is_eligible_accepts_client_when_rules_are_met():
    client = {"age": 35, "housing": "yes"}
    rules = {"min_age": 18, "requires_housing_loan": True}

    assert is_eligible(client, rules) is True


def test_get_eligible_offers_filters_offers_by_rules():
    client = {"age": 25, "housing": "no"}
    offers = [
        {"arm_id": "0", "eligibility_rules": {"min_age": 18, "requires_housing_loan": False}},
        {"arm_id": "1", "eligibility_rules": {"min_age": 30, "requires_housing_loan": True}},
    ]

    eligible = get_eligible_offers(client, offers)

    assert [offer["arm_id"] for offer in eligible] == ["0"]


def test_expected_reward_returns_prior_times_reward_value():
    offer = {"synthetic_conversion_prior": 0.4, "reward_value": 20}

    assert expected_reward(offer) == 8.0
