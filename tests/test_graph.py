from graph.nodes import check_gemini_output, generate_groq_output, generate_rag_prompt
from graph.router import llm_router
from graph.state import AgentState


def test_generate_rag_prompt_builds_state_with_prompt(monkeypatch):
    def fake_build_rag_prompt(prompt):
        return f"prompt:{prompt}"

    monkeypatch.setattr("graph.nodes.build_rag_prompt", fake_build_rag_prompt)
    state = {"prompt": "pergunta"}

    result = generate_rag_prompt(state)

    assert result["rag_prompt"] == "prompt:pergunta"


def test_check_gemini_output_routes_to_groq_when_needed():
    state = {"output": "Trying Groq..."}

    assert check_gemini_output(state) == "call_groq"


def test_llm_router_routes_to_finish_for_successful_response():
    state = {"output": "Resposta pronta"}

    assert llm_router(state) == "finish"


def test_generate_groq_output_returns_error_message_when_call_fails(monkeypatch):
    def fake_ask_groq(prompt):
        raise RuntimeError("fail")

    monkeypatch.setattr("graph.nodes.ask_groq", fake_ask_groq)
    state = {"rag_prompt": "prompt"}

    result = generate_groq_output(state)

    assert "Error occurred while generating Groq output." in result["output"]
