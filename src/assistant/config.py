"""Configuração do assistente, carregada de variáveis de ambiente.

O provedor padrão é ``offline``: o pipeline roda inteiro sem chave de API,
o que permite testes em CI e demonstração sem custo. Para usar um LLM real,
basta definir ``ASSISTANT_PROVIDER`` e as credenciais correspondentes no .env.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # carrega .env se python-dotenv estiver instalado (é dependência do projeto)
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - ambiente sem dotenv ainda funciona
    pass

# Raiz do repositório (…/src/assistant/config.py -> sobe 3 níveis)
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class AssistantConfig:
    """Parâmetros do assistente LLM + RAG."""

    # Provedor do LLM: "offline" | "azure" | "openai" | "anthropic"
    provider: str = field(default_factory=lambda: os.getenv("ASSISTANT_PROVIDER", "offline"))
    model: str = field(default_factory=lambda: os.getenv("ASSISTANT_MODEL", "gpt-4o-mini"))
    temperature: float = field(default_factory=lambda: float(os.getenv("ASSISTANT_TEMPERATURE", "0.2")))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("ASSISTANT_MAX_TOKENS", "700")))

    # Caminhos dos dados (com defaults coerentes com o repositório)
    policies_dir: Path = field(
        default_factory=lambda: Path(os.getenv("ASSISTANT_POLICIES_DIR", str(REPO_ROOT / "data" / "policies")))
    )
    metrics_path: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "ASSISTANT_METRICS_PATH",
                str(REPO_ROOT / "data" / "experiments" / "etapa3" / "metrics_timeseries.parquet"),
            )
        )
    )
    catalog_path: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "ASSISTANT_CATALOG_PATH",
                str(REPO_ROOT / "data" / "kaggle" / "synthetic_enrichment" / "offer_catalog.json"),
            )
        )
    )

    # RAG
    top_k: int = field(default_factory=lambda: int(os.getenv("ASSISTANT_TOP_K", "3")))

    # Versão da política do assistente (entra nos logs auditáveis)
    policy_version: str = field(default_factory=lambda: os.getenv("ASSISTANT_POLICY_VERSION", "assistant-0.1.0"))

    def describe(self) -> str:
        return (
            f"provider={self.provider} model={self.model} "
            f"top_k={self.top_k} policy_version={self.policy_version}"
        )
