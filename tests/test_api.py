import pytest

import api.app as api_app
from api.app import app

CLIENTE_VALIDO = {
    "age": 20,
    "balance": 1500,
    "housing": "no",
    "loan": "no",
    "job": "manager",
    "marital": "married",
    "education": "no",
}


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


def test_predict_requires_client_fields(client):
    response = client.post("/predict", json={"age": 20})

    assert response.status_code == 400


def test_predict_returns_answer(client, monkeypatch):
    monkeypatch.setattr(
        api_app,
        "run_prompt",
        lambda prompt: {"prompt": prompt, "output": "resposta fake do LLM", "rag_prompt": "..."},
    )

    response = client.post("/predict", json=CLIENTE_VALIDO)

    assert response.status_code == 200
    body = response.get_json()
    assert body["answer"] == "resposta fake do LLM"
    assert body["cliente"] == CLIENTE_VALIDO


def test_predict_returns_500_when_pipeline_raises(client, monkeypatch):
    def _raise(prompt):
        raise RuntimeError("falha simulada")

    monkeypatch.setattr(api_app, "run_prompt", _raise)

    response = client.post("/predict", json=CLIENTE_VALIDO)

    assert response.status_code == 500
