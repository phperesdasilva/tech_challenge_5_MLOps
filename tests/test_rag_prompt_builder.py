from rag.prompt_builder import build_rag_prompt


def test_build_rag_prompt_includes_context_and_user_prompt(monkeypatch):
    def fake_retrieve_context(prompt):
        return [{"metadata": {"source": "doc1"}, "chunk": "Contexto exemplo"}]

    monkeypatch.setattr("rag.prompt_builder.retrieve_context", fake_retrieve_context)

    prompt = build_rag_prompt("Qual é a resposta?")

    assert "Você é um assistente virtual especialista" in prompt
    assert "Qual é a resposta?" in prompt
    assert "Contexto exemplo" in prompt


def test_build_rag_prompt_handles_no_context(monkeypatch):
    monkeypatch.setattr("rag.prompt_builder.retrieve_context", lambda prompt: [])

    prompt = build_rag_prompt("Pergunta sem contexto")

    assert "Nenhum contexto relevante encontrado" in prompt
    assert "Pergunta sem contexto" in prompt
