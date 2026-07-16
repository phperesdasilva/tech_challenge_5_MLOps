"""Orquestrador do assistente LLM + RAG."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import guardrails, prompts
from .config import AssistantConfig
from .llm_client import LLMClient
from .retriever import PolicyRetriever, RetrievalResult


@dataclass
class AssistantResponse:
    task: str
    answer: str
    sources: List[str] = field(default_factory=list)
    audit: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExperimentAssistant:
    def __init__(self, config: Optional[AssistantConfig] = None) -> None:
        self.config = config or AssistantConfig()
        self.llm = LLMClient(self.config)
        self._retriever: Optional[PolicyRetriever] = None
        self._catalog: Optional[Dict[str, dict]] = None

    @property
    def retriever(self) -> PolicyRetriever:
        if self._retriever is None:
            self._retriever = PolicyRetriever(self.config.policies_dir, k1=1.5, b=0.75)
        return self._retriever

    def _load_catalog(self) -> Dict[str, dict]:
        if self._catalog is None:
            self._catalog = {}
            p = self.config.catalog_path
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                for offer in data.get("offers", []):
                    self._catalog[str(offer["arm_id"])] = offer
        return self._catalog

    def _audit_base(self, task: str) -> Dict[str, Any]:
        return {
            "task": task,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "provider": self.config.provider,
            "model": self.config.model,
            "policy_version": self.config.policy_version,
        }

    def summarize_experiment(self, metrics_path: Optional[Path] = None) -> AssistantResponse:
        import pandas as pd

        path = Path(metrics_path or self.config.metrics_path)
        audit = self._audit_base("summarize_experiment")
        audit["metrics_path"] = str(path)

        if not path.exists():
            return AssistantResponse(
                task="summarize_experiment",
                answer=(
                    f"Não encontrei o arquivo de métricas em {path}. "
                    "Rode o experimento (run-experiment) para gerá-lo."
                ),
                audit=audit,
            )

        df = pd.read_parquet(path)
        finals = df.sort_values("step").groupby("policy").tail(1).set_index("policy")
        lines = []
        for policy, row in finals.iterrows():
            lines.append(
                f"- {policy}: recompensa acumulada={row['cumulative_reward']:.0f}, "
                f"regret acumulado={row['cumulative_regret']:.0f}, "
                f"conversão final={row['conversion_rate']:.4f}"
            )
        context = (
            f"Passos simulados: {int(df['step'].max())}. Políticas comparadas: "
            f"{', '.join(finals.index)}.\n" + "\n".join(lines)
        )
        user_prompt = f"<contexto>\n{context}\n</contexto>\n\nResuma o experimento."
        answer = guardrails.scrub_output(self.llm.complete(prompts.SUMMARIZE_EXPERIMENT, user_prompt))
        audit["policies"] = list(finals.index)
        return AssistantResponse(task="summarize_experiment", answer=answer, audit=audit)

    def answer_policy_question(self, question: str) -> AssistantResponse:
        audit = self._audit_base("answer_policy_question")
        audit["question"] = question

        guard = guardrails.check_input(question)
        if not guard.allowed:
            audit["blocked"] = True
            return AssistantResponse(task="answer_policy_question", answer=guard.reason, audit=audit)

        hits: List[RetrievalResult] = self.retriever.search(question, k=self.config.top_k)
        sources = sorted({h.chunk.source for h in hits})
        audit["retrieved"] = [
            {"source": h.chunk.source, "title": h.chunk.title, "score": round(h.score, 3)} for h in hits
        ]

        if not hits:
            return AssistantResponse(
                task="answer_policy_question",
                answer="Não há base documental sobre isso nas políticas sintéticas disponíveis.",
                sources=[],
                audit=audit,
            )

        context = "\n\n".join(
            f"[Fonte: {h.chunk.source} — {h.chunk.title}]\n{h.chunk.snippet(500)}" for h in hits
        )
        user_prompt = f"<contexto>\n{context}\n</contexto>\n\n<pergunta>{question}</pergunta>"
        answer = guardrails.scrub_output(self.llm.complete(prompts.ANSWER_POLICY_QUESTION, user_prompt))
        return AssistantResponse(task="answer_policy_question", answer=answer, sources=sources, audit=audit)

    def explain_decision(self, decision: Dict[str, Any]) -> AssistantResponse:
        audit = self._audit_base("explain_decision")
        catalog = self._load_catalog()
        arm_id = str(decision.get("arm_id", ""))
        offer = catalog.get(arm_id, {})
        offer_name = offer.get("offer_name", f"arm {arm_id}")
        offer_type = offer.get("offer_type", "")
        reason_codes = decision.get("reason_codes", [])

        audit.update({
            "arm_id": arm_id,
            "offer_name": offer_name,
            "offer_type": offer_type,
            "reason_codes": reason_codes,
            "bandit_policy_version": decision.get("policy_version"),
            "context": decision.get("context", {}),
        })

        rag_query = f"suitability elegibilidade {offer_name} {offer_type}"
        hits = self.retriever.search(rag_query, k=2)
        sources = sorted({h.chunk.source for h in hits})
        policy_context = "\n".join(f"[{h.chunk.source}] {h.chunk.snippet(260)}" for h in hits)

        context = (
            f"Braço selecionado: {offer_name} (arm {arm_id}, tipo {offer_type}).\n"
            f"Reason codes: {', '.join(reason_codes) if reason_codes else 'n/d'}.\n"
            f"Contexto de decisão: {json.dumps(decision.get('context', {}), ensure_ascii=False)}.\n"
            f"Políticas relacionadas:\n{policy_context}"
        )
        user_prompt = f"<contexto>\n{context}\n</contexto>\n\nExplique a decisão."
        answer = guardrails.scrub_output(self.llm.complete(prompts.EXPLAIN_DECISION, user_prompt))
        answer += guardrails.human_in_the_loop_note(offer_type)

        return AssistantResponse(task="explain_decision", answer=answer, sources=sources, audit=audit)
