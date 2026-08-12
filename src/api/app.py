"""
API Flask para recomendação de oferta bancária a um cliente via LLM.

Endpoints:
    GET  /health   - confirma que a API subiu.
    POST /predict  - recebe um JSON com os dados do cliente e devolve a oferta
                     recomendada pelo pipeline RAG + LLM.
"""

import os

from dotenv import load_dotenv
from flask import Flask, jsonify, request

from llm.llm_main import run_prompt

# Carrega variáveis de ambiente
load_dotenv()

# Campos com os dados do cliente esperados no body de /predict.
CAMPOS_CLIENTE = ["age", "balance", "housing", "loan", "job", "marital", "education"]

app = Flask(__name__)


def montar_prompt_cliente(client: dict) -> str:
    """Monta o prompt em linguagem natural enviado ao LLM a partir dos dados do cliente."""
    return (
        "Com base no perfil de cliente abaixo, recomende a melhor oferta bancária "
        "disponível para ele e justifique a recomendação.\n\n"
        "Dados do cliente:\n"
        f"- Idade: {client['age']}\n"
        f"- Saldo em conta: {client['balance']}\n"
        f"- Possui financiamento imobiliário (housing): {client['housing']}\n"
        f"- Possui empréstimo pessoal (loan): {client['loan']}\n"
        f"- Profissão: {client['job']}\n"
        f"- Estado civil: {client['marital']}\n"
        f"- Escolaridade: {client['education']}\n"
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    client = request.get_json(silent=True)
    if not isinstance(client, dict):
        return jsonify({"erro": "corpo da requisição deve ser um JSON com os dados do cliente"}), 400

    campos_ausentes = [campo for campo in CAMPOS_CLIENTE if campo not in client]
    if campos_ausentes:
        return jsonify({"erro": f"campos obrigatórios ausentes: {', '.join(campos_ausentes)}"}), 400

    prompt = montar_prompt_cliente(client)

    try:
        resultado = run_prompt(prompt)
    except Exception as e:
        return jsonify({"erro": f"falha ao consultar o LLM: {e}"}), 500

    return jsonify(
        {
            "cliente": client,
            "answer": resultado["output"],
        }
    )


if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "5000"))
    app.run(host=host, port=port)
