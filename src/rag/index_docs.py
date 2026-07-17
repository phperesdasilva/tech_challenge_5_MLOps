import os
import re
import json
from pathlib import Path
from dotenv import load_dotenv
import faiss
import numpy as np

from rag.embeddings import embedding_model

load_dotenv()

# Definição dos caminhos das variáveis de ambiente
VECTOR_STORE_DIR = Path(os.getenv("VECTOR_STORE_DIR", "data/rag"))
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH")
METADATA_PATH = os.getenv("METADATA_PATH")

# Lista de arquivos Markdown a serem consumidos
MD_FILES = [
    os.getenv("TS_METRICS_REPORT_PATH"),
    os.getenv("ARM_COUNTS_REPORT_BL_PATH"),
    os.getenv("ARM_COUNTS_REPORT_TS_PATH"),
    os.getenv("OFFER_CATALOG_REPORT_PATH")
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

    # 1. Tenta identificar se o chunk inicia com um título Markdown
    section_match = re.search(r'^(?:#|##|###)\s+(.+)$', chunk_text, re.MULTILINE)
    if section_match:
        metadata["secao_detectada"] = section_match.group(1).strip()

    # 2. Busca por menções a IDs de braço (ex: "id_braco: 0", "Braço 1", "id_braco: '2'")
    arm_match = re.search(r'(?:id_braco|id do braço|braço)\s*[:\-]?\s*\'?(\d+)\'?', chunk_text, re.IGNORECASE)
    if arm_match:
        metadata["id_braco_mencionado"] = int(arm_match.group(1))

    # 3. Mapeamento simples de palavras-chave do domínio para auxiliar filtros rápidos
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

    # 1. Indexação exclusiva dos arquivos Markdown (.md)
    for filepath in MD_FILES:
        if not filepath:
            continue
        path = Path(filepath)
        if not path.exists():
            print(f"⚠️ Aviso: Arquivo {filepath} não encontrado. Pulando...")
            continue

        print(f"Lendo e dividindo o documento: {path.name}...")
        text = path.read_text(encoding="utf-8")
        chunks = chunk_markdown(text)

        # Define uma tag amigável baseada no nome do arquivo para compor o ID do chunk
        # ex: "metrics_timeseries_report" -> "ts_metrics"
        doc_tag = path.stem.replace("_report", "").replace("report_", "")

        for idx, chunk in enumerate(chunks):
            # Extrai os metadados de dentro do próprio bloco de texto
            extracted_info = extract_metadata_from_chunk(chunk)

            chunks_to_index.append(chunk)
            metadata_to_index.append({
                "source": path.name,
                # Resolve a sobreposição de IDs: ex: "ts_metrics_chunk_0", "offer_catalog_chunk_0"
                "chunk_id": f"{doc_tag}_chunk_{idx}",
                "text": chunk,
                "metadata": {
                    "tipo_documento": "relatorio_markdown",
                    "id_braco": extracted_info["id_braco_mencionado"],
                    "secao": extracted_info["secao_detectada"],
                    "tags_conteudo": extracted_info["palavras_chave"]
                }
            })

    if not chunks_to_index:
        print("❌ Nenhum texto encontrado nos arquivos Markdown para indexar.")
        return

    # 2. Geração e gravação dos vetores no FAISS
    print(f"Gerando embeddings para {len(chunks_to_index)} trechos...")
    embeddings = np.array(embedding_model.encode(chunks_to_index)).astype('float32')

    dimension = embeddings.shape[1]

    # Garante que a pasta de destino exista
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    print(f"Salvando o índice FAISS em: {VECTOR_STORE_PATH}")
    faiss.write_index(index, str(VECTOR_STORE_PATH))

    # 3. Grava os metadados estruturados ricos
    print(f"Salvando metadados enriquecidos em: {METADATA_PATH}")
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata_to_index, f, indent=4, ensure_ascii=False)

    print("✅ Banco de dados vetorial FAISS e metadados atualizados com sucesso!")
