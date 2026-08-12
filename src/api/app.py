"""
API Flask para recomendação de oferta (braço do bandit) para um cliente.

Endpoints:
    GET  /health   - confirma que a API subiu e as policies já foram treinadas.
    POST /predict  - recebe um JSON de cliente e devolve a oferta recomendada.
    POST /ask-llm  - recebe um JSON com "prompt" e devolve a resposta do pipeline
                     RAG + LLM (mesmo pipeline do comando "ask-llm" do CLI).

Regra de roteamento entre as policies:
    - Se o JSON do cliente trouxer TODOS os campos de contexto usados pelo
      BankContextEncoder (idade, saldo, housing, loan, profissão, estado
      civil, escolaridade) -> usa LinUCBPolicy (contextual).
    - Se faltar qualquer um desses campos -> usa ThompsonSamplingPolicy
      (não-contextual, só precisa de "age" para checar elegibilidade).
"""

import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, request

from bandit.catalog import get_eligible_offers, load_catalog
from bandit.features import BINARY_COLS, CATEGORICAL_COLS, NUMERIC_COLS, BankContextEncoder
from bandit.policies import LinUCBPolicy, ThompsonSamplingPolicy
from bandit.simulator import BASE_DATE, DEFAULT_BANK_PATH, run_simulation
from llm.llm_main import run_prompt
from rag.retriever import retrieve_context

# Carrega variáveis de ambiente
load_dotenv()

SEED = int(os.getenv("SEED", "42"))

# Campos que o BankContextEncoder precisa para montar o vetor de contexto do LinUCB.
CAMPOS_CONTEXTO = NUMERIC_COLS + BINARY_COLS + CATEGORICAL_COLS


def treinar_policies():
    """Treina ThompsonSamplingPolicy e LinUCBPolicy em memória, do zero.

    Reaproveita bandit.simulator.run_simulation() -- o mesmo motor de simulação
    usado pelo CLI (ThompsonSamplingSimulator/LinUCBSimulator) -- para "replay"
    da base histórica de clientes e treinar as policies antes de servir
    qualquer requisição.
    """
    offers = load_catalog()
    id_bracos = [oferta["id_braco"] for oferta in offers]
    df_clientes = pd.read_parquet(DEFAULT_BANK_PATH)

    encoder = BankContextEncoder(df_clientes)

    thompson_policy = ThompsonSamplingPolicy(id_bracos)
    run_simulation(
        thompson_policy,
        df_clientes,
        offers,
        np.random.default_rng(SEED),
        base_date=BASE_DATE,
    )

    linucb_policy = LinUCBPolicy(id_bracos, dim=encoder.dim)
    run_simulation(
        linucb_policy,
        df_clientes,
        offers,
        np.random.default_rng(SEED),
        base_date=BASE_DATE,
        encoder=encoder,
    )

    return offers, encoder, thompson_policy, linucb_policy


# Warm-up: roda uma única vez, quando o módulo é importado (subida do processo Flask).
OFFERS, ENCODER, THOMPSON_POLICY, LINUCB_POLICY = treinar_policies()

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "ofertas_carregadas": len(OFFERS)
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    client = request.get_json(silent=True)
    if not isinstance(client, dict):
        return jsonify({"erro": "corpo da requisição deve ser um JSON com os dados do cliente"}), 400
    if "age" not in client:
        return jsonify({"erro": "campo obrigatório ausente: 'age'"}), 400

    eligible_offers = get_eligible_offers(client, OFFERS)
    if not eligible_offers:
        return jsonify({"erro": "nenhuma oferta elegível para este cliente"}), 404

    eligible_ids = [oferta["id_braco"] for oferta in eligible_offers]

    tem_contexto_completo = all(
        campo in client and client[campo] is not None for campo in CAMPOS_CONTEXTO
    )

    if tem_contexto_completo:
        contexto = ENCODER.encode(client)
        id_braco_escolhido = LINUCB_POLICY.select_arm(eligible_ids, context=contexto)
        policy_usada = LINUCB_POLICY.name()
    else:
        id_braco_escolhido = THOMPSON_POLICY.select_arm(eligible_ids)
        policy_usada = THOMPSON_POLICY.name()

    oferta_escolhida = next(o for o in eligible_offers if o["id_braco"] == id_braco_escolhido)

    return jsonify(
        {
            "policy_used": policy_usada,
            "id_braco": oferta_escolhida["id_braco"],
            "nome_oferta": oferta_escolhida["nome_oferta"],
            "tipo_oferta": oferta_escolhida.get("tipo_oferta"),
            "descricao": oferta_escolhida.get("descricao"),
        }
    )


@app.route("/ask-llm", methods=["POST"])
def ask_llm():
    body = request.get_json(silent=True)
    if not isinstance(body, dict) or not body.get("prompt"):
        return jsonify({"erro": "campo obrigatório ausente: 'prompt'"}), 400

    prompt = body["prompt"]

    try:
        contexto = retrieve_context(prompt)
        resultado = run_prompt(prompt)
    except Exception as e:
        return jsonify({"erro": f"falha ao consultar o LLM: {e}"}), 500

    return jsonify(
        {
            "prompt": prompt,
            "answer": resultado["output"],
            "sources": [
                {
                    "source": item["metadata"]["source"],
                    "score": item["score"],
                    "tipo_documento": item["metadata"]["tipo_documento"],
                    "id_braco": item["metadata"]["id_braco"],
                }
                for item in contexto
            ],
        }
    )


if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "5000"))
    app.run(host=host, port=port)
