import pytest

import api.app as api_app
from api.app import CAMPOS_CONTEXTO, app


@pytest.fixture()
def client():
    return app.test_client()


def test_health_returns_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_predict_requires_json_body(client):
    response = client.post("/predict", data="not-json", content_type="text/plain")

    assert response.status_code == 400


def test_predict_requires_age_field(client):
    response = client.post("/predict", json={"housing": "no"})

    assert response.status_code == 400


def test_predict_returns_404_when_client_is_ineligible_for_any_offer(client):
    response = client.post("/predict", json={"age": 10, "housing": "no"})

    assert response.status_code == 404


def test_predict_uses_thompson_sampling_when_context_is_incomplete(client):
    response = client.post("/predict", json={"age": 30, "housing": "no"})

    assert response.status_code == 200
    body = response.get_json()
    assert body["policy_used"] == "ThompsonSamplingPolicy"
    assert "id_braco" in body


def test_predict_uses_linucb_when_full_context_is_provided(client):
    payload = {
        "age": 40,
        "balance": 1500,
        "housing": "no",
        "loan": "no",
        "job": "admin.",
        "marital": "married",
        "education": "tertiary",
    }
    assert set(CAMPOS_CONTEXTO) <= set(payload)

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.get_json()["policy_used"] == "LinUCBPolicy"


def test_predict_falls_back_to_thompson_when_one_context_field_is_missing(client):
    payload = {
        "age": 40,
        "balance": 1500,
        "housing": "no",
        "loan": "no",
        "job": "admin.",
        "marital": "married",
        # education ausente de propósito
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.get_json()["policy_used"] == "ThompsonSamplingPolicy"


def test_ask_llm_requires_prompt_field(client):
    response = client.post("/ask-llm", json={})

    assert response.status_code == 400


def test_ask_llm_returns_answer_and_sources(client, monkeypatch):
    monkeypatch.setattr(
        api_app,
        "retrieve_context",
        lambda prompt: [
            {
                "score": 0.9,
                "chunk": "trecho recuperado",
                "metadata": {"source": "doc.md", "tipo_documento": "faq", "id_braco": "0"},
            }
        ],
    )
    monkeypatch.setattr(
        api_app,
        "run_prompt",
        lambda prompt: {"prompt": prompt, "output": "resposta fake do LLM", "rag_prompt": "..."},
    )

    response = client.post("/ask-llm", json={"prompt": "qual a taxa do cartão premium?"})

    assert response.status_code == 200
    body = response.get_json()
    assert body["answer"] == "resposta fake do LLM"
    assert body["sources"] == [
        {"source": "doc.md", "score": 0.9, "tipo_documento": "faq", "id_braco": "0"}
    ]


def test_ask_llm_returns_500_when_pipeline_raises(client, monkeypatch):
    def _raise(prompt):
        raise RuntimeError("falha simulada")

    monkeypatch.setattr(api_app, "retrieve_context", _raise)

    response = client.post("/ask-llm", json={"prompt": "qualquer coisa"})

    assert response.status_code == 500
