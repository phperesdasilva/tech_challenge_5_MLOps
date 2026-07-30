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
        False, "--all", help="Gera todos os relatórios de todos os experimentos (Thompson Sampling + LinUCB)."
    ),
    oc: bool = typer.Option(
        False, "--oc", help="Gera o relatório do catálogo de ofertas."
    ),
    all_linucb_reports: bool = typer.Option(
        False, "--linucb", help="Gera todos os relatórios do LinUCB de uma vez."
    ),
    all_thompson_sampling_reports: bool = typer.Option(
        False, "--thompson-sampling", help="Gera todos os relatórios do Thompson Sampling de uma vez."
    )
):
    """
    Gera os relatórios executivos em Markdown utilizando LLM.
    Você deve passar pelo menos uma das opções: --all, --tsmetrics, --acbl, --acts, --oc, --all linucb.
    Para relatórios específicos do LinUCB, use o comando "generate-linucb-report".
    """
    if not (all_reports or oc or all_linucb_reports or all_thompson_sampling_reports):
        typer.echo("[Erro] Você precisa selecionar pelo menos um relatório para gerar.")
        raise typer.Exit(code=1)

    if all_reports:
        typer.echo("Iniciando geração de todos os relatórios (Thompson Sampling + LinUCB)...")
        cli.generate_all_experiment_reports()
        return

    if oc:
        typer.echo("Iniciando geração do relatório do Catálogo de Ofertas...")
        cli.generate_report_offer_catalog()


    if all_linucb_reports:
        typer.echo("Iniciando geração de todos os relatórios do LinUCB...")
        cli.generate_all_linucb_reports()
        return

    if all_thompson_sampling_reports:
        typer.echo("Iniciando geração de todos os relatórios do Thompson Sampling...")
        cli.generate_all_thompson_sampling_reports()
        return
    

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
