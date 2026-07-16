"""CLI do assistente. Entrada registrada em pyproject como ``run-assistant``."""
from __future__ import annotations

import argparse
import json
import sys
from typing import List

from .assistant import ExperimentAssistant
from .config import AssistantConfig


def _print(resp, as_json: bool) -> None:
    if as_json:
        print(json.dumps(resp.to_dict(), ensure_ascii=False, indent=2))
        return
    print(resp.answer)
    if resp.sources:
        print("\nFontes:", ", ".join(resp.sources))
    print("\n--- log auditável ---")
    print(json.dumps(resp.audit, ensure_ascii=False, indent=2))


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run-assistant", description="Assistente LLM + RAG (Datathon)")
    parser.add_argument("--json", action="store_true", help="saída em JSON")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("summarize", help="resume o experimento mais recente")

    p_ask = sub.add_parser("ask", help="pergunta sobre políticas (RAG)")
    p_ask.add_argument("question", help="pergunta em linguagem natural")

    p_exp = sub.add_parser("explain", help="explica uma decisão do bandit")
    p_exp.add_argument("--arm", required=True, help="arm_id selecionado")
    p_exp.add_argument("--reason", action="append", default=[], help="reason code (pode repetir)")
    p_exp.add_argument("--policy-version", default=None, help="versão da política do bandit")

    args = parser.parse_args(argv)

    config = AssistantConfig()
    assistant = ExperimentAssistant(config)
    print(f"# Assistente ({config.describe()})\n", file=sys.stderr)

    if args.command == "summarize":
        resp = assistant.summarize_experiment()
    elif args.command == "ask":
        resp = assistant.answer_policy_question(args.question)
    elif args.command == "explain":
        resp = assistant.explain_decision(
            {
                "arm_id": args.arm,
                "reason_codes": args.reason,
                "policy_version": args.policy_version,
                "context": {},
            }
        )
    else:  # pragma: no cover
        parser.error("comando desconhecido")
        return 2

    _print(resp, args.json)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
