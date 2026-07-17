from graph.builder import build_graph

app = build_graph()

def run_llm(prompt=None):

    if prompt is None:
        prompt = input("Digite seu prompt (ou 'sair' para encerrar): ")
        print(f"\nPrompt enviado para o modelo...\n")
    if prompt.lower() == 'sair':
        print("Encerrando a interação com o modelo.")
        return


    output = app.invoke({"prompt": prompt})

    print(f"""
⬜⬜⬜⬜⬜⬜⬜

{output['output']}

⬜⬜⬜⬜⬜⬜⬜
""")
