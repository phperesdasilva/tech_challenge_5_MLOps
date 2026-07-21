from graph.builder import build_graph

app = build_graph()

def run_llm(prompt=None):

    while True:

        prompt = input("Digite seu prompt (ou 'sair' para encerrar): ")

        if prompt.lower() == 'sair':
            print("Encerrando a interação com o modelo.")
            break


        output = app.invoke({"prompt": prompt})

        print(f"""
    ⬜⬜⬜⬜⬜⬜⬜

    {output['output']}

    ⬜⬜⬜⬜⬜⬜⬜
    """)
