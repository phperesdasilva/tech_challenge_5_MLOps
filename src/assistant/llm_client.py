"""Cliente de LLM provider-agnóstico.

Provedores suportados (via ``ASSISTANT_PROVIDER``):
  - ``offline`` (padrão): gerador determinístico que monta a resposta a partir do
    contexto. Não requer chave nem rede — ideal para CI e demonstração.
  - ``azure``: Azure OpenAI (alvo de produção do desafio, Etapa 6).
  - ``openai``: OpenAI.
  - ``anthropic``: Anthropic Claude.

Os SDKs reais são importados de forma preguiçosa e ficam em dependência opcional
(extra ``llm``), então o módulo instala e roda sem nenhum deles.
"""
from __future__ import annotations

import os
import re
import textwrap

from .config import AssistantConfig


class LLMClient:
    def __init__(self, config: AssistantConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------ API
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        provider = self.config.provider.lower()
        if provider == "offline":
            return self._offline(system_prompt, user_prompt)
        if provider == "azure":
            return self._azure(system_prompt, user_prompt)
        if provider == "openai":
            return self._openai(system_prompt, user_prompt)
        if provider == "anthropic":
            return self._anthropic(system_prompt, user_prompt)
        raise ValueError(f"Provedor desconhecido: {self.config.provider}")

    # -------------------------------------------------------------- offline
    def _offline(self, system_prompt: str, user_prompt: str) -> str:
        """Resposta determinística construída a partir do CONTEXTO do prompt.

        Não é um modelo de linguagem: extrai o bloco de contexto e o reorganiza
        em uma resposta legível, deixando claro que está em modo offline. Serve
        para validar todo o fluxo (retrieval, montagem de contexto, guardrails,
        logging) antes de plugar um LLM real.
        """
        context = self._extract_context(user_prompt)
        question = self._extract_question(user_prompt)
        header = "[modo offline — sem LLM real]"
        if context:
            body = textwrap.shorten(" ".join(context.split()), width=900, placeholder="…")
            answer = (
                f"{header}\nCom base no contexto fornecido:\n\n{body}\n\n"
                "Observação: este texto é um resumo direto do contexto. Conecte um "
                "provedor real (azure/openai/anthropic) para uma resposta em linguagem natural."
            )
        else:
            answer = f"{header} Não há contexto suficiente para responder a: {question or user_prompt}"
        return answer

    @staticmethod
    def _extract_context(user_prompt: str) -> str:
        m = re.search(r"<contexto>(.*?)</contexto>", user_prompt, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""

    @staticmethod
    def _extract_question(user_prompt: str) -> str:
        m = re.search(r"<pergunta>(.*?)</pergunta>", user_prompt, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""

    # ---------------------------------------------------------------- azure
    def _azure(self, system_prompt: str, user_prompt: str) -> str:  # pragma: no cover - requer credenciais
        from openai import AzureOpenAI

        client = AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01"),
        )
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", self.config.model)
        resp = client.chat.completions.create(
            model=deployment,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return resp.choices[0].message.content or ""

    # --------------------------------------------------------------- openai
    def _openai(self, system_prompt: str, user_prompt: str) -> str:  # pragma: no cover - requer credenciais
        from openai import OpenAI

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        resp = client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return resp.choices[0].message.content or ""

    # ------------------------------------------------------------ anthropic
    def _anthropic(self, system_prompt: str, user_prompt: str) -> str:  # pragma: no cover - requer credenciais
        import anthropic

        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        resp = client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")
