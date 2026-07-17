import os
import json
from pathlib import Path
from dotenv import load_dotenv
import faiss
import numpy as np

# Importa o modelo de embeddings que você já possui estruturado no seu projeto
from rag.embeddings import embedding_model

load_dotenv()

VECTOR_STORE_DIR = Path("data/rag")
VECTOR_STORE_PATH = VECTOR_STORE_DIR / "vector_store.faiss"
METADATA_PATH = VECTOR_STORE_DIR / "vector_store_metadata.json"

MD_FILES = [
    os.getenv("TS_METRICS_REPORT_PATH"),
    os.getenv("ARM_COUNTS_REPORT_BL_PATH"),
    os.getenv("ARM_COUNTS_REPORT_TS_PATH")
]

def chunk_markdown(text, chunk_size=500, overlap=50):
    """
    Função simples para dividir o texto em partes (chunks) menores por parágrafos.
    Mantém o contexto do relatório legível para a busca semântica.
    """
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = []
    current_length = 0

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        paragraph_len = len(paragraph)
        if current_length + paragraph_len > chunk_size and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            # Mantém uma pequena sobreposição (overlap) para não quebrar o contexto
            current_chunk = current_chunk[-1:] if len(current_chunk) > 1 else current_chunk
            current_length = sum(len(p) for p in current_chunk)

        current_chunk.append(paragraph)
        current_length += paragraph_len

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks

def index_documents():
    chunks_to_index = []
    metadados_to_index = []

    for filepath in MD_FILES:
        path = Path(filepath)
        if not path.exists():
            print(f"⚠️ Aviso: Arquivo {filepath} não encontrado. Pulando...")
            continue

        print(f"Lendo e dividindo o documento: {path.name}...")
        text = path.read_text(encoding="utf-8")
        chunks = chunk_markdown(text)

        for idx, chunk in enumerate(chunks):
            chunks_to_index.append(chunk)
            metadados_to_index.append({
                "source": path.name,
                "chunk_id": idx,
                "text": chunk
            })

    if not chunks_to_index:
        print("❌ Nenhum texto encontrado para indexar.")
        return

    print(f"Gerando embeddings para {len(chunks_to_index)} trechos...")
    embeddings = np.array(embedding_model.encode(chunks_to_index)).astype('float32')

    dimension = embeddings.shape[1]

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    print(f"Salvando o índice FAISS em: {VECTOR_STORE_PATH}")
    faiss.write_index(index, str(VECTOR_STORE_PATH))

    print(f"Salvando metadados em: {METADATA_PATH}")
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadados_to_index, f, indent=4, ensure_ascii=False)

    print("✅ Banco de dados vetorial FAISS criado e atualizado com sucesso!")
