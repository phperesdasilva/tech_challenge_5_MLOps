# evaluation/ragas_eval.py
"""Avaliação do pipeline RAG com RAGAS — 4 métricas obrigatórias.
Referência: Es et al. (2024) — RAGAS: Automated Evaluation of Retrieva
l
Augmented Generation. https://arxiv.org/abs/2309.15217
"""
import json
import logging
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
answer_relevancy,
context_precision,
context_recall,
faithfulness,
)

logger = logging.getLogger(__name__)

def evaluate_rag_pipeline(
    golden_set_path: str,
    rag_fn,
    ) -> dict[str, float]:

    """Avalia pipeline RAG contra golden set.
    Args:
    golden_set_path: Caminho para JSON com golden set.
    rag_fn: Função que recebe query e retorna
    (answer, contexts).
    Returns:
    Dicionário com 4 métricas RAGAS.
    """

    with open(golden_set_path) as f:
        golden_set = json.load(f)
    # Gera respostas do pipeline
    results = []

    for item in golden_set:
        answer, contexts = rag_fn(item["query"])
        results.append({
        "question": item["query"],
        "answer": answer,
        "contexts": contexts,
        "ground_truth": item["expected_answer"],
        })

    dataset = Dataset.from_list(results)

    # Avaliação RAGAS — 4 métricas obrigatórias
    scores = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    metrics = {
        "faithfulness": scores["faithfulness"],
        "answer_relevancy": scores["answer_relevancy"],
        "context_precision": scores["context_precision"],
        "context_recall": scores["context_recall"],
        }
    
    logger.info("RAGAS scores: %s", metrics)
    return metrics