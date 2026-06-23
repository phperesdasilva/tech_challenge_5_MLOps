import typer
from cli.CLI import CLI

app = typer.Typer(
    help="Tech Challenge 5 - MLOps",
    rich_markup_mode="rich",
    no_args_is_help=True
)

cli = CLI()

@app.command(name="run-eda")
def run_eda():
    """
    Executa a análise exploratória de dados (EDA) no dataset especificado.
    """
    cli.run_eda()

@app.command(name="generate-events")
def generate_events():
    """
    Gera eventos sintéticos para simular um ambiente de Multi-Armed Bandit.
    """
    cli.generate_events()

def main():
    app(prog_name="project")

if __name__ == "__main__":
    main()
