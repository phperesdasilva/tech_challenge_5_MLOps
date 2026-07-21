from rag.index_docs import chunk_markdown, extract_metadata_from_chunk


def test_chunk_markdown_splits_text_into_multiple_parts():
    text = "Parágrafo 1\n\nParágrafo 2\n\nParágrafo 3"

    chunks = chunk_markdown(text, chunk_size=30)

    assert len(chunks) >= 2
    assert any("Parágrafo 1" in chunk for chunk in chunks)


def test_chunk_markdown_returns_single_chunk_when_text_fits():
    text = "Texto curto"

    chunks = chunk_markdown(text, chunk_size=200)

    assert chunks == [text]


def test_extract_metadata_from_chunk_detects_section_and_arm_id():
    chunk_text = "## Resultados\nO braço 3 foi analisado"

    metadata = extract_metadata_from_chunk(chunk_text)

    assert metadata["secao_detectada"] == "Resultados"
    assert metadata["id_braco_mencionado"] == 3


def test_extract_metadata_from_chunk_returns_empty_keywords_when_missing():
    chunk_text = "Texto sem padrões especiais"

    metadata = extract_metadata_from_chunk(chunk_text)

    assert metadata["palavras_chave"] == []
