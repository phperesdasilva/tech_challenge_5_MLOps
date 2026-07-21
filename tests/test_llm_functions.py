import pytest

from llm.gemini_model import ask_gemini
from llm.groq_model import ask_groq


class DummyResponse:
    def __init__(self, text):
        self.text = text


class DummyMessage:
    def __init__(self, content):
        self.content = content


class DummyChoice:
    def __init__(self, content):
        self.message = DummyMessage(content)


class DummyCompletionResponse:
    def __init__(self, content):
        self.choices = [DummyChoice(content)]


class DummyGeminiClient:
    def __init__(self, response_text):
        self.models = type("Models", (), {"generate_content": lambda self, **kwargs: DummyResponse(response_text)})()


class DummyGroqClient:
    def __init__(self, response_text):
        self.chat = type("Chat", (), {"completions": type("Completions", (), {"create": lambda self, **kwargs: DummyCompletionResponse(response_text)})()})()


def test_ask_gemini_returns_text_from_client(monkeypatch):
    monkeypatch.setattr("llm.gemini_model.gemini_client", DummyGeminiClient("saída gemini"))

    assert ask_gemini("prompt") == "saída gemini"


def test_ask_groq_returns_text_from_client(monkeypatch):
    monkeypatch.setattr("llm.groq_model.groq_client", DummyGroqClient("saída groq"))

    assert ask_groq("prompt") == "saída groq"
