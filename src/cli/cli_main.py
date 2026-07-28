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

@app.command(name="run-linucb")
def run_linucb():
    cli.run_linucb()

@app.command(name="generate-report")
def generate_report(
    all_reports: bool = typer.Option(
        False, "--all", help="Gera todos os 3 relatórios disponíveis de uma vez."
    ),
    tsmetrics: bool = typer.Option(
        False, "--tsmetrics", help="Gera o relatório de progressão das métricas do Thompson Sampling."
    ),
    acbl: bool = typer.Option(
        False, "--acbl", help="Gera o relatório de contagem de execuções de braços da política Baseline."
    ),
    acts: bool = typer.Option(
        False, "--acts", help="Gera o relatório de contagem de execuções de braços do Thompson Sampling."
    ),
    oc: bool = typer.Option(
        False, "--oc", help="Gera o relatório do catálogo de ofertas."
    )
):
    """
    Gera os relatórios executivos em Markdown utilizando LLM.
    Você deve passar pelo menos uma das opções: --all, --tsmetrics, --acbl, --acts, --oc.
    """
    if not (all_reports or tsmetrics or acbl or acts or oc):
        typer.echo("[Erro] Você precisa selecionar pelo menos um relatório para gerar.")
        raise typer.Exit(code=1)

    if all_reports:
        typer.echo("Iniciando geração de todos os relatórios...")
        cli.generate_all_reports()
        return

    if tsmetrics:
        typer.echo("Iniciando geração do relatório de métricas do Thompson Sampling...")
        cli.generate_report_ts_metrics()

    if acbl:
        typer.echo("Iniciando geração do relatório da Baseline...")
        cli.generate_report_arm_counts_bl()

    if acts:
        typer.echo("Iniciando geração do relatório do Thompson Sampling (Arm Counts)...")
        cli.generate_report_arm_counts_ts()

    if oc:
        typer.echo("Iniciando geração do relatório do Catálogo de Ofertas...")
        cli.generate_report_offer_catalog()

@app.command(name="index-documents")
def index_documents():
    """
    Indexa os documentos Markdown gerados para busca semântica.
    """
    typer.echo("Iniciando indexação dos documentos...")
    cli.index_documents()

@app.command(name="retrieve-context")
def retrieve_context(
    query: str = typer.Argument(
        ...,
        help="A frase, pergunta ou termo que você deseja buscar no banco de dados vetorial."
    ),
    top_k: int = typer.Option(
        4,
        "--top-k", "-k",
        help="Quantidade de trechos de documentos mais similares a serem retornados."
    )
):
    """
    Realiza uma busca semântica no banco de dados vetorial FAISS e retorna os trechos mais relevantes.
    """
    cli.retrieve_context(query=query, top_k=top_k)

@app.command(name="build-rag-prompt")
def build_rag_prompt(
    prompt: str = typer.Argument(
        ...,
        help="A frase, pergunta ou termo que você deseja buscar no banco de dados vetorial."
    )
):
    """
    Constrói o prompt RAG com base na consulta do usuário e no contexto recuperado.
    """
    rag_prompt = cli.build_rag_prompt(prompt)
    typer.echo(rag_prompt)

@app.command(name="ask-llm")
def llm_run(
    prompt: str = typer.Argument(
        None,
        help="Prompt a ser enviado para o modelo LLM. Se não fornecido, será solicitado interativamente."
    )
):
    """
    Executa o loop de interação com o modelo LLM.
    """
    cli.ask_llm(prompt=prompt)

def main():
    app(prog_name="project")

if __name__ == "__main__":
    main()
