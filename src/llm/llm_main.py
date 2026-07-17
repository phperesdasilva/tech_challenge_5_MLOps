from graph.builder import build_graph

app = build_graph()

def run_llm(prompt=None):

    if prompt is None:
        prompt = input("Digite seu prompt (ou 'sair' para encerrar): ")
    if prompt.lower() == 'sair':
        print("Encerrando a interação com o modelo.")
        return

    print(f"Prompt enviado para o modelo...")

    output = app.invoke({"prompt": prompt})

    print(f"""
⬜⬜⬜⬜⬜⬜⬜

{output['output']}

⬜⬜⬜⬜⬜⬜⬜
""")
