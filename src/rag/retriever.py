import faiss
import os
import json
from functools import lru_cache

from dotenv import load_dotenv
from rag.embeddings import get_embedding_model

load_dotenv()

METADATA_PATH = os.getenv(
    "METADATA_PATH",
    "data/rag/vector_store_metadata.json",
)


@lru_cache(maxsize=1)
def _get_index():
    return faiss.read_index("data/rag/vector_store.faiss")


@lru_cache(maxsize=1)
def _get_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def retrieve_context(prompt, top_k=4):
    print(f"\nRetrieving context...\n")
    index = _get_index()
    metadata = _get_metadata()
    prompt_vector = get_embedding_model().encode([prompt], convert_to_numpy=True).astype('float32')

    scores, positions = index.search(prompt_vector, top_k)

    results = []

    for score, pos in zip(scores[0], positions[0]):
        # Filtro de segurança para índices inválidos retornados pelo FAISS
        if pos == -1 or pos >= len(metadata):
            continue


        metadados_originais = metadata[pos].get("metadata", {})

        results.append({
            "score": float(score),
            "chunk": metadata[pos]["text"],
            "metadata": {
                "source": metadata[pos]["source"],
                "chunk_id": metadata[pos]["chunk_id"],
                "tipo_documento": metadados_originais.get("tipo_documento"),
                "id_braco": metadados_originais.get("id_braco"),
                "secao": metadados_originais.get("secao"),
                "tags_conteudo": metadados_originais.get("tags_conteudo", [])
            }
        })

    return results
