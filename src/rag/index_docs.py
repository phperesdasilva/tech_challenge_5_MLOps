import os
import re
import json
from pathlib import Path
from dotenv import load_dotenv
import faiss
import numpy as np

load_dotenv()

from rag.embeddings import embedding_model
import rag.paths as paths



VECTOR_STORE_DIR = Path(os.getenv("VECTOR_STORE_DIR", "data/rag"))
VECTOR_STORE_PATH = Path(
    os.getenv("VECTOR_STORE_PATH", str(VECTOR_STORE_DIR / "vector_store.faiss"))
)
METADATA_PATH = Path(
    os.getenv("METADATA_PATH", str(VECTOR_STORE_DIR / "vector_store_metadata.json"))
)

MD_FILES = [
    # Thompson Sampling
    paths.TS_METRICS_REPORT_PATH,
    paths.ARM_COUNTS_REPORT_BL_PATH,
    paths.ARM_COUNTS_REPORT_TS_PATH,
    paths.OFFER_CATALOG_REPORT_PATH,
    # LinUCB
    paths.LINUCB_METRICS_REPORT_PATH,
    paths.ARM_COUNTS_REPORT_LINUCB_PATH,
    paths.ARM_BY_JOB_REPORT_PATH,
    paths.ARM_BY_EDUCATION_REPORT_PATH,
    paths.ARM_BY_POUTCOME_REPORT_PATH,
    paths.ARM_BY_AGE_GROUP_REPORT_PATH,
    paths.GOLDEN_SET_PATH
]

def chunk_markdown(text, chunk_size=1500):
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
            current_chunk = current_chunk[-1:] if len(current_chunk) > 1 else current_chunk
            current_length = sum(len(p) for p in current_chunk)

        current_chunk.append(paragraph)
        current_length += paragraph_len

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks

def extract_metadata_from_chunk(chunk_text):
    """
    Analisa o conteúdo do bloco de texto e extrai metadados úteis para o RAG,
    como títulos de seções, IDs de braço mencionados e palavras-chave.
    """
    metadata = {
        "secao_detectada": None,
        "id_braco_mencionado": None,
        "palavras_chave": []
    }

    section_match = re.search(r'^(?:#|##|###)\s+(.+)$', chunk_text, re.MULTILINE) #busca titulo do md
    if section_match:
        metadata["secao_detectada"] = section_match.group(1).strip()

    arm_match = re.search(r'(?:id_braco|id do braço|braço)\s*[:\-]?\s*\'?(\d+)\'?', chunk_text, re.IGNORECASE) #busca id do braço
    if arm_match:
        metadata["id_braco_mencionado"] = int(arm_match.group(1))

    keywords_map = {
        "thompson": "Thompson Sampling",
        "baseline": "Baseline",
        "regret": "Arrependimento/Regret",
        "conversão": "Conversão",
        "recompensa": "Recompensa/Reward",
        "oferta": "Catálogo de Ofertas",
        "idade": "Regra de Elegibilidade"
    }
    for key, val in keywords_map.items():
        if key in chunk_text.lower():
            metadata["palavras_chave"].append(val)

    return metadata

def index_documents():
    chunks_to_index = []
    metadata_to_index = []

    for filepath in MD_FILES:

        VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

        path = Path(filepath)

        print(f"Lendo e dividindo o documento: {path.name}...")
        text = path.read_text(encoding="utf-8")
        chunks = chunk_markdown(text)

        # Define uma tag amigável baseada no nome do arquivo para compor o ID do chunk
        # ex: "metrics_timeseries_report" -> "ts_metrics"
        doc_tag = path.stem.replace("_report", "").replace("report_", "")

        for idx, chunk in enumerate(chunks):
            extracted_info = extract_metadata_from_chunk(chunk)

            chunks_to_index.append(chunk)
            metadata_to_index.append({
                "source": path.name,
                "chunk_id": f"{doc_tag}_chunk_{idx}",
                "text": chunk,
                "metadata": {
                    "tipo_documento": "relatorio_markdown",
                    "id_braco": extracted_info["id_braco_mencionado"],
                    "secao": extracted_info["secao_detectada"],
                    "tags_conteudo": extracted_info["palavras_chave"]
                }
            })


    print(f"Gerando embeddings para {len(chunks_to_index)} trechos...")
    embeddings = np.array(embedding_model.encode(chunks_to_index)).astype('float32')

    dimension = embeddings.shape[1]


    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    print(f"Salvando o índice FAISS em: {VECTOR_STORE_PATH}")
    faiss.write_index(index, str(VECTOR_STORE_PATH))

    print(f"Salvando metadados enriquecidos em: {METADATA_PATH}")
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata_to_index, f, indent=4, ensure_ascii=False)

    print("✅ Banco de dados vetorial FAISS e metadados atualizados com sucesso!")
