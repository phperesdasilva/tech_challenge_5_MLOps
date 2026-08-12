import os
from functools import lru_cache

import mlflow

from bandit.tracking import configure_mlflow
from graph.builder import build_graph


@lru_cache(maxsize=1)
def _get_app():
    return build_graph()


def run_llm(prompt=None):
    configure_mlflow(os.getenv("MLFLOW_EXPERIMENT_LLM", "LLM_RAG"))
    mlflow.gemini.autolog()
    mlflow.groq.autolog()

    while True:

        prompt = input("Digite seu prompt (ou 'sair' para encerrar): ")

        if prompt.lower() == 'sair':
            print("Encerrando a interação com o modelo.")
            break


        output = run_prompt(prompt)

        print(f"""
    ⬜⬜⬜⬜⬜⬜⬜

    {output['output']}

    ⬜⬜⬜⬜⬜⬜⬜
    """)


@mlflow.trace(name="run_llm")
def run_prompt(prompt: str) -> dict:
    return _get_app().invoke({"prompt": prompt})
