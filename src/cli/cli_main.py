import typer
from typing import List
from cli.CLI import CLI

app = typer.Typer(
    help="Tech Challenge 5 - MLOps",
    rich_markup_mode="rich",
    no_args_is_help=True
)

cli = CLI()

@app.command(name="run-eda")
def run_eda():
    """Executa a análise exploratória de dados (EDA) no dataset especificado."""
    cli.run_eda()

@app.command(name="generate-events")
def generate_events():
    """Gera eventos sintéticos para simular um ambiente de Multi-Armed Bandit."""
    cli.generate_events()

@app.command(name="run-thompson-sampling")
def run_thompson_sampling():
    """Executa a simulação do algoritmo Thompson Sampling e salva os resultados."""
    cli.run_thompson_sampling()

@app.command(name="summarize")
def summarize():
    """Resume os resultados do experimento de bandit mais recente."""
    cli.summarize()

@app.command(name="ask")
def ask(question: str = typer.Argument(..., help="Pergunta sobre políticas internas")):
    """Responde perguntas sobre políticas via RAG (sem chave de API)."""
    cli.ask(question)

@app.command(name="explain")
def explain(
    arm: str = typer.Option(..., "--arm", help="arm_id selecionado pelo bandit"),
    reason: List[str] = typer.Option([], "--reason", help="reason code (pode repetir)"),
):
    """Explica em linguagem natural a decisão do bandit para um braço."""
    cli.explain(arm, reason)

def main():
    app(prog_name="project")

if __name__ == "__main__":
    main()
