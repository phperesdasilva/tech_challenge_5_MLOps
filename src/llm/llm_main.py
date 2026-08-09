import os

import mlflow

from bandit.tracking import configure_mlflow
from graph.builder import build_graph

app = build_graph()

def run_llm(prompt=None):
    configure_mlflow(os.getenv("MLFLOW_EXPERIMENT_LLM", "LLM_RAG"))
    mlflow.gemini.autolog()
    mlflow.groq.autolog()

    while True:

        prompt = input("Digite seu prompt (ou 'sair' para encerrar): ")

        if prompt.lower() == 'sair':
            print("Encerrando a interação com o modelo.")
            break


        output = _run_graph(prompt)

        print(f"""
    ⬜⬜⬜⬜⬜⬜⬜

    {output['output']}

    ⬜⬜⬜⬜⬜⬜⬜
    """)


@mlflow.trace(name="run_llm")
def _run_graph(prompt: str) -> dict:
    return app.invoke({"prompt": prompt})
