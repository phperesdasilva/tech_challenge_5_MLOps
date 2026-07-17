from rag.retriever import retrieve_context

def build_rag_prompt(prompt):

    context = retrieve_context(prompt)

    if not context:
        context_str = "Nenhum contexto relevante encontrado para a consulta."
    else:
        context_str = "\n\n".join(
            [f"Contexto {i+1} (Score: {item['score']:.4f}):\n{item['chunk']}" for i, item in enumerate(context)]
        )

    return f"""
Responda à seguinte pergunta utilizando o contexto fornecido. Se o contexto não for suficiente, responda da melhor forma possível.
Seja cauteloso para não inventar informações. Se não souber a resposta, diga "Não sei".
O contexto relevante encontrado para a consulta é o seguinte:
{context_str}
"""
