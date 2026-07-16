"""Camada de recuperação (RAG) sobre o corpus sintético de políticas.

Implementação baseline com **BM25** em Python puro (sem dependências externas),
o que torna o retrieval reprodutível e executável em CI sem chave de API.

Evolução natural (documentada no README do módulo): trocar o BM25 por embeddings
(ex.: Azure OpenAI embeddings) e um índice vetorial (Azure AI Search ou FAISS),
mantendo a mesma interface ``search()``.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Chunk:
    """Trecho recuperável de um documento de política."""

    source: str  # nome do arquivo de origem
    title: str  # título/seção do trecho
    text: str  # conteúdo do trecho

    def snippet(self, limit: int = 320) -> str:
        t = " ".join(self.text.split())
        return t if len(t) <= limit else t[:limit].rsplit(" ", 1)[0] + "…"


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float


_TOKEN_RE = re.compile(r"[\wÀ-ÿ]+", re.UNICODE)


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _split_into_chunks(path: Path) -> List[Chunk]:
    """Divide um .md em trechos por cabeçalho de seção (##)."""
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    chunks: List[Chunk] = []
    current_title = path.stem
    buffer: List[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            chunks.append(Chunk(source=path.name, title=current_title, text=body))

    for line in lines:
        if line.startswith("## "):
            flush()
            current_title = line.lstrip("#").strip()
            buffer = []
        elif line.startswith("# "):
            current_title = line.lstrip("#").strip()
        else:
            buffer.append(line)
    flush()
    return chunks


class PolicyRetriever:
    """Índice BM25 simples sobre os documentos de política."""

    def __init__(self, policies_dir: Path, k1: float = 1.5, b: float = 0.75) -> None:
        self.policies_dir = Path(policies_dir)
        self.k1 = k1
        self.b = b
        self.chunks: List[Chunk] = []
        self._doc_tokens: List[List[str]] = []
        self._doc_freqs: List[dict] = []
        self._df: dict = {}
        self._avgdl: float = 0.0
        self._build()

    def _build(self) -> None:
        if not self.policies_dir.exists():
            raise FileNotFoundError(f"Pasta de políticas não encontrada: {self.policies_dir}")
        for md in sorted(self.policies_dir.glob("*.md")):
            self.chunks.extend(_split_into_chunks(md))
        if not self.chunks:
            raise ValueError(f"Nenhum trecho indexável em {self.policies_dir}")

        for c in self.chunks:
            toks = _tokenize(f"{c.title} {c.text}")
            self._doc_tokens.append(toks)
            freqs: dict = {}
            for t in toks:
                freqs[t] = freqs.get(t, 0) + 1
            self._doc_freqs.append(freqs)
            for t in set(toks):
                self._df[t] = self._df.get(t, 0) + 1

        lengths = [len(t) for t in self._doc_tokens]
        self._avgdl = sum(lengths) / len(lengths)

    def _idf(self, term: str) -> float:
        n = len(self.chunks)
        df = self._df.get(term, 0)
        # IDF do BM25 (Robertson), com piso para evitar valores negativos
        return max(0.0, math.log((n - df + 0.5) / (df + 0.5) + 1.0))

    def search(self, query: str, k: int = 3) -> List[RetrievalResult]:
        q_terms = _tokenize(query)
        scored: List[RetrievalResult] = []
        for i, freqs in enumerate(self._doc_freqs):
            dl = len(self._doc_tokens[i])
            score = 0.0
            for term in q_terms:
                if term not in freqs:
                    continue
                f = freqs[term]
                idf = self._idf(term)
                denom = f + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                score += idf * (f * (self.k1 + 1)) / denom
            if score > 0:
                scored.append(RetrievalResult(chunk=self.chunks[i], score=score))
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:k]
