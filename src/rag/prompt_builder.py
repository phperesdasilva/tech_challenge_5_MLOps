from rag.retriever import retrieve_context

def build_rag_prompt(prompt):

    context = retrieve_context(prompt)

    if not context:
        context_str = "Nenhum contexto relevante encontrado para a consulta."
    else:
        context_str = "\n\n".join(
            [f"--- Documento {i+1} (Origem: {item['metadata']['source']}) ---\n{item['chunk']}"
             for i, item in enumerate(context)]
        )

    return f"""Você é um assistente virtual especialista no projeto.

[INSTRUÇÕES]
Responda à pergunta do usuário utilizando o contexto fornecido abaixo.
Se o contexto não for suficiente ou a resposta não estiver documentada, diga explicitamente que não possui documentos sobre o assunto e responda da melhor forma possível com o seu conhecimento, sem inventar dados.
Seja extremamente cauteloso para não alucinar informações.
Se a pergunta não estiver clara, peça esclarecimentos.

[CONTEXTO DE SUPORTE]
{context_str}

[PERGUNTA DO USUÁRIO]
{prompt}
"""
