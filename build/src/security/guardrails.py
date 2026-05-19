# src/security/guardrails.py
"""Guardrails de segurança para input e output do agente.
Referência: OWASP Top 10 for LLM Applications (2025)
https://owasp.org/www-project-top-10-for-large-language-mo
del-applications/
"""
import logging
import re
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

logger = logging.getLogger(__name__)

class InputGuardrail:
    """Valida e sanitiza input do usuário antes de enviar ao LLM."""
    # Padrões comuns de prompt injection
    INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"you\s+are\s+now\s+a",
    r"system:\s*",
    r"<\|im_start\|>",
    r"\[INST\]",
    r"forget\s+(everything|all|your\s+instructions)",
    ]

    def __init__(self, allowed_topics: list[str] | None = None):
        self.allowed_topics = allowed_topics or []
        self._compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS]
    
    def validate(self, user_input: str) -> tuple[bool, str]:
        """Valida input do usuário.
        Args:
        user_input: Texto do usuário.
        Returns:
        Tupla (is_valid, reason).
        """
        # Check 1: Prompt injection detection
        for pattern in self._compiled_patterns:
            if pattern.search(user_input):
                logger.warning("Prompt injection detectado: %s", user_input[:100])
            return False, "Input bloqueado: padrão suspeito detectado."
        
        # Check 2: Tamanho máximo (evitar context stuffing)
        if len(user_input) > 4096:
            return False, "Input bloqueado: excede tamanho máximo (4096 chars)."
        
        return True, "OK"
    
class OutputGuardrail:
    """Valida e sanitiza output do LLM antes de retornar ao usuário."""
    def __init__(self, language: str = "pt"):
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        self.language = language

    def sanitize(self, llm_output: str) -> str:
        """Remove PII do output do LLM.
        Args:
        llm_output: Texto gerado pelo LLM.
        Returns:
        Texto sanitizado.
        """
        results = self.analyzer.analyze(
        text=llm_output,
        language=self.language,
        entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "BR_CPF"],
        )

        if results:
            logger.warning("PII detectado no output: %d entidades", len(results))
            anonymized = self.anonymizer.anonymize(
            text=llm_output,
            analyzer_results=results,
            )

            return anonymized.text
        
        return llm_output
    