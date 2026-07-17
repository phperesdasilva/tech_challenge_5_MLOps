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
        # Se o FAISS retornar -1, significa que não encontrou vizinhos suficientes
        if pos == -1 or pos >= len(metadata):
            continue

        # O FAISS retorna distâncias L2. Se você quiser filtrar por um score de corte (ex: MIN_SCORE)
        # Note que para distância L2, quanto MENOR o valor, mais similar é o documento.
        # Se preferir usar um filtro simples, pode descomentar a linha abaixo:
        #if float(score) > MIN_SCORE: continue

        # Recupera as informações corretas usando o índice retornado 'pos' mapeado no JSON de metadados
        results.append({
            "score": float(score),
            "chunk": metadata[pos]["text"],        # O texto do bloco de dados original
            "metadata": {
                "source": metadata[pos]["source"], # Nome do arquivo de origem (ex: baseline_count_report.md)
                "chunk_id": metadata[pos]["chunk_id"]
            }
        })

    return results
