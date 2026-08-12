"""
API Flask para recomendação de oferta bancária a um cliente via LLM.

Endpoints:
    GET  /health   - confirma que a API subiu.
    POST /predict  - recebe um JSON com os dados do cliente e devolve a oferta
                     recomendada pelo pipeline RAG + LLM.
    GET  /apidocs  - interface gráfica interativa (Swagger UI) para documentação da API.
"""

import os

from dotenv import load_dotenv
from flasgger import Swagger, swag_from
from flask import Flask, jsonify, request

from llm.llm_main import run_prompt

# Carrega variáveis de ambiente
load_dotenv()

# Campos com os dados do cliente esperados no body de /predict.
CAMPOS_CLIENTE = ["age", "balance", "housing", "loan", "job", "marital", "education"]

app = Flask(__name__)

# Configuração da documentação Swagger
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": "apispec",
            "route": "/apispec.json",
            "rule_filter": lambda rule: True,
            "model_filter": lambda tag: True,
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/apidocs/",
}

template = {
    "swagger": "2.0",
    "info": {
        "title": "API de Recomendação de Oferta Bancária (LLM)",
        "description": "API REST para consulta e sugestão de produtos bancários baseada no perfil do cliente.",
        "version": "1.0.0",
    },
}

swagger = Swagger(app, config=swagger_config, template=template)


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
    """
    Confirma a saúde da aplicação.
    ---
    tags:
      - Monitoramento
    responses:
      200:
        description: API está ativa e operacional.
        schema:
          type: object
          properties:
            status:
              type: string
              example: "ok"
    """
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Gera uma recomendação bancária para o cliente usando RAG + LLM.
    ---
    tags:
      - Recomendação
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - age
            - balance
            - housing
            - loan
            - job
            - marital
            - education
          properties:
            age:
              type: integer
              example: 35
              description: Idade do cliente.
            balance:
              type: number
              example: 12500.50
              description: Saldo atual em conta.
            housing:
              type: string
              example: "yes"
              description: "Possui financiamento imobiliário (yes/no)."
            loan:
              type: string
              example: "no"
              description: "Possui empréstimo pessoal (yes/no)."
            job:
              type: string
              example: "technician"
              description: Profissão ou ocupação principal.
            marital:
              type: string
              example: "married"
              description: Estado civil.
            education:
              type: string
              example: "tertiary"
              description: Nível de escolaridade.
    responses:
      200:
        description: Oferta bancária recomendada com sucesso.
        schema:
          type: object
          properties:
            cliente:
              type: object
            answer:
              type: string
              example: "Com base no seu saldo e estabilidade, recomendamos a Cartão de Crédito Black com cashback..."
      400:
        description: Requisição inválida ou campos obrigatórios ausentes.
      500:
        description: Erro interno ao consultar o modelo LLM.
    """
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
