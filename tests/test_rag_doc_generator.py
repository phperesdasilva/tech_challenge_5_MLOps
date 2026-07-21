from pathlib import Path

import pytest

from rag.doc_generator import generate_report, generate_report_with_groq


class DummyResponse:
    def __init__(self, text):
        self.text = text


class DummyChoices:
    def __init__(self, text):
        self.message = type("Message", (), {"content": text})()


class DummyChatCompletion:
    def __init__(self, text):
        self.choices = [DummyChoices(text)]


class DummyGroqClient:
    def __init__(self, text):
        self.text = text
        self.chat = type("Chat", (), {"completions": type("Completions", (), {"create": lambda self, **kwargs: DummyChatCompletion(text)})()})()


class DummyGeminiClient:
    def __init__(self, text):
        self.text = text
        self.files = type("Files", (), {"upload": lambda self, file, config=None: {"path": file}})()
        self.models = type("Models", (), {"generate_content": lambda self, **kwargs: DummyResponse(text)})()


def test_generate_report_with_groq_reads_source_and_returns_content(tmp_path, monkeypatch):
    source = tmp_path / "data.txt"
    source.write_text("conteúdo de teste", encoding="utf-8")

    dummy_client = DummyGroqClient("relatório groq")
    monkeypatch.setattr("rag.doc_generator.groq_client", dummy_client)

    result = generate_report_with_groq("prompt", str(source))

    assert result == "relatório groq"


def test_generate_report_writes_output_to_disk_when_gemini_succeeds(tmp_path, monkeypatch):
    source = tmp_path / "data.csv"
    source.write_text("col1,col2\n1,2", encoding="utf-8")
    report_path = tmp_path / "report.md"

    monkeypatch.setattr("rag.doc_generator.gemini_client", DummyGeminiClient("relatório gemini"))

    generate_report("prompt", str(source), str(report_path))

    assert report_path.read_text(encoding="utf-8") == "relatório gemini"


def test_generate_report_falls_back_to_groq_when_gemini_fails(tmp_path, monkeypatch):
    source = tmp_path / "data.json"
    source.write_text('{"a": 1}', encoding="utf-8")
    report_path = tmp_path / "report.md"

    class FailingGeminiClient:
        files = type("Files", (), {"upload": lambda self, file, config=None: (_ for _ in ()).throw(RuntimeError("fail"))})()
        models = type("Models", (), {"generate_content": lambda self, **kwargs: (_ for _ in ()).throw(RuntimeError("fail"))})()

    monkeypatch.setattr("rag.doc_generator.gemini_client", FailingGeminiClient())
    monkeypatch.setattr("rag.doc_generator.groq_client", DummyGroqClient("fallback"))

    generate_report("prompt", str(source), str(report_path))

    assert report_path.read_text(encoding="utf-8") == "fallback"
