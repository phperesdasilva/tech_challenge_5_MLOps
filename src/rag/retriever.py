import faiss
import os
import json

from dotenv import load_dotenv
from rag.embeddings import embedding_model

index = faiss.read_index("data/rag/vector_store.faiss")

load_dotenv()

MIN_SCORE = 0.4
METADATA_PATH = os.getenv("METADATA_PATH")


with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

def retrieve_context(prompt, top_k = 2):
    prompt_vector = embedding_model.encode([prompt], convert_to_numpy=True).astype('float32')

    scores, positions = index.search(prompt_vector, top_k)

    results =[]

    for score, pos in zip(scores[0], positions[0]):
        if pos == -1 or pos >= len(metadata):
            continue

        results.append({
            "score": float(score),
            "chunk": metadata[pos]["text"],
            "metadata": {
                "source": metadata[pos]["source"],
                "chunk_id": metadata[pos]["chunk_id"]
            }
        })

    return results
