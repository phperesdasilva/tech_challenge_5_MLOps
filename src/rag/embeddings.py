from functools import lru_cache


@lru_cache(maxsize=1)
def get_embedding_model():
    """Carrega o modelo de embeddings sob demanda (o import de sentence_transformers/torch é pesado)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('all-MiniLM-L6-v2')
