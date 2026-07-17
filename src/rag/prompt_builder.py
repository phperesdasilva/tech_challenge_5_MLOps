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
Responda ao prompt utilizando o contexto fornecido. Se o contexto não for suficiente, responda da melhor forma possível.
Seja cauteloso para não inventar informações.
Se a resposta não estiver documentada, diga que não tem documentos sobre isso e responda por si mesmo, sem inventar informações.
Se a pergunta não estiver clara, peça esclarecimentos.
O contexto relevante encontrado para a consulta é o seguinte:
{context_str}
"""
