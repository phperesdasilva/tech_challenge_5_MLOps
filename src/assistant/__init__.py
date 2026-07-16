"""Assistente LLM + RAG da plataforma de experimentação adaptativa.

Capacidades (Datathon): resumir experimentos, recuperar políticas sintéticas (RAG)
e explicar decisões do multi-armed bandit, com guardrails e logs auditáveis.
"""
from .assistant import AssistantResponse, ExperimentAssistant
from .config import AssistantConfig
from .retriever import PolicyRetriever

__all__ = [
    "ExperimentAssistant",
    "AssistantResponse",
    "AssistantConfig",
    "PolicyRetriever",
]
