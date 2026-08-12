# Tech Challenge 5 - MLOps

Projeto de MLOps para recomendação de ofertas bancárias. O repositório reúne a preparação dos dados, a geração de eventos sintéticos, a simulação de políticas de bandit, o pipeline RAG para relatórios e uma API Flask para consulta de recomendações assistidas por LLM.

## O Que Este Projeto Faz

O fluxo atual do projeto é este:

1. Preparar e limpar a base Bank Marketing.
2. Gerar eventos sintéticos com recompensas e atrasos.
3. Simular políticas de recomendação com Thompson Sampling e LinUCB.
4. Gerar relatórios Markdown a partir dos resultados dos experimentos, do catálogo de ofertas e das políticas de negócio.
5. Indexar esses documentos em FAISS para busca semântica e uso em RAG.
6. Expor uma API Flask para recomendar uma oferta com base no perfil do cliente.

## Estrutura Atual

- `src/api/`: API Flask com `/health`, `/predict` e Swagger UI em `/apidocs`.
- `src/bandit/`: catálogo de ofertas, ambiente, features, métricas, políticas e tracking.
- `src/cli/`: comando principal `project` com as rotas operacionais do projeto.
- `src/eda/`: notebooks e scripts de preparação/análise dos dados.
- `src/event_generator/`: geração dos eventos sintéticos e recompensas atrasadas.
- `src/experiments/`: simulações de Thompson Sampling e LinUCB.
- `src/graph/`: grafo e roteamento de estado usados pelo fluxo conversacional.
- `src/llm/`: integrações com Gemini, Groq e orquestração do prompt.
- `src/rag/`: geração de documentos, embeddings, indexação, paths e recuperação de contexto.
- `data/`: insumos, artefatos sintéticos, relatórios e índices gerados.
- `docs/model-card.md`: resumo do modelo e das suas limitações.
- `tests/`: testes automatizados.

## Requisitos

- Python `>=3.12.3`
- Dependências principais: `numpy`, `pandas`, `pyarrow`, `python-dotenv`, `flask`, `flasgger`, `faiss-cpu`, `sentence-transformers`, `google-genai`, `groq`, `langgraph`, `typer`
- Dependências opcionais de desenvolvimento: `pytest`, `pytest-cov`, `pre-commit`, `black`, `isort`
- Dependências opcionais de MLOps: `kaggle`, `mlflow`

## Instalação

Crie e ative o ambiente virtual e depois instale o projeto em modo editável:

```bash
pip install -e .
```

Para instalar os extras de desenvolvimento:

```bash
pip install -e ".[dev]"
```

Para instalar os extras de MLOps:

```bash
pip install -e ".[mlops]"
```

Para instalar tudo junto:

```bash
pip install -e ".[dev,mlops]"
```

## Dados E Artefatos

O projeto espera, por padrão, os seguintes dados e artefatos:

- `data/kaggle/processed/clean_bank.parquet`
- `data/kaggle/synthetic_enrichment/offer_catalog.json`
- `data/kaggle/synthetic_enrichment/offer_events.csv`
- `data/kaggle/synthetic_enrichment/delayed_rewards.csv`
- `data/golden_set/evaluation_cases.jsonl`
- `data/policies/faq-ofertas.md`
- `data/policies/governanca-dados.md`
- `data/policies/politica-comunicacao.md`
- `data/policies/politica-elegibilidade.md`
- `data/policies/politica-suitability.md`

Os relatórios e índices gerados pelo RAG ficam em `data/rag/` e os resultados dos experimentos em `data/experiments/`.

## Como Executar

### Preparar os dados

Abra os notebooks em `src/eda/` para explorar a base e preparar os arquivos intermediários, ou execute:

```bash
project run-eda
```

O script `src/eda/run_eda.py` executa a preparação principal dos dados.

### Gerar eventos sintéticos

Use o comando de console:

```bash
project generate-events
```

### Rodar as simulações

Thompson Sampling:

```bash
project run-thompson-sampling
```

LinUCB:

```bash
project run-linucb
```

### Gerar relatórios em Markdown

O comando `generate-report` aceita combinações de flags:

- `project generate-report --oc` gera o relatório do catálogo de ofertas.
- `project generate-report --linucb` gera todos os relatórios do LinUCB.
- `project generate-report --thompson-sampling` gera todos os relatórios do Thompson Sampling.
- `project generate-report --all` gera todos os relatórios de uma vez.

### Indexar documentos no RAG

Depois de gerar os relatórios, indexe os Markdown com:

```bash
project index-documents
```

### Consultar o contexto recuperado

```bash
project retrieve-context "sua pergunta aqui"
```

### Construir prompt RAG

```bash
project build-rag-prompt "sua pergunta aqui"
```

### Falar com o LLM

```bash
project ask-llm
```

### Subir a API

```bash
project start-api
```

Para a interface de experimentos do MLflow:

```bash
project start-mlflow-ui
```

A API expõe:

- `GET /health`
- `POST /predict`
- `GET /apidocs/`

## Saídas Principais

- `data/experiments/thompson_sampling/`: métricas e contagens do experimento Thompson Sampling.
- `data/experiments/linucb/`: métricas e contagens do experimento LinUCB.
- `data/rag/`: relatórios Markdown e índice FAISS do RAG.
- `data/golden_set/`: casos de validação do conjunto dourado.

## Variáveis De Ambiente

As variáveis abaixo podem ser definidas em `.env` quando necessário:

- `API_HOST` e `API_PORT`: host e porta da API Flask.
- `VECTOR_STORE_DIR`, `VECTOR_STORE_PATH` e `METADATA_PATH`: local do índice e dos metadados do RAG.
- `TS_METRICS_PATH`, `ARM_COUNTS_BL_PATH`, `ARM_COUNTS_TS_PATH`, `OFFER_CATALOG_PATH`: entradas usadas na geração de relatórios do Thompson Sampling.
- `LINUCB_METRICS_SUMMARY_PATH`, `ARM_COUNTS_LINUCB_PATH`, `ARM_BY_JOB_PATH`, `ARM_BY_EDUCATION_PATH`, `ARM_BY_POUTCOME_PATH`, `ARM_BY_AGE_GROUP_PATH`: entradas usadas na geração de relatórios do LinUCB.
- `POLICIES_DIR`: diretório base da pasta de políticas.
- `DEFAULT_BANK_PATH` e `DEFAULT_CATALOG_PATH`: caminhos esperados pelo fluxo de simulação.
- `BASE_DATE`, `DELAY_SCALE_DAYS`, `SEED`, `THOMPSON_ALPHA0`, `THOMPSON_BETA0`, `UCB1_EXPLORATION_BONUS`: parâmetros da simulação.
- `OPENAI_API_KEY`, `GOOGLE_API_KEY`, ou equivalentes usados pelos clientes de LLM, quando aplicável.

## Limitações Conhecidas

- A abordagem segue majoritariamente um desenho não contextual nas políticas clássicas de bandit.
- Os relatórios RAG dependem dos arquivos Markdown já gerados em `data/rag/`.
- A API de recomendação depende da disponibilidade do LLM configurado no ambiente.

## Testes

Execute os testes com:

```bash
pytest
```

## Golden Set — Casos de Teste (Etapa 4)

O arquivo [`data/golden_set/evaluation_cases.jsonl`](data/golden_set/evaluation_cases.jsonl) reúne 20 casos de teste — 5 conversões reais observadas na simulação para cada uma das 4 ofertas do catálogo —, mostrando qual oferta foi recomendada, para qual cliente, e se a decisão faz sentido em relação ao valor esperado do catálogo. Os casos das ofertas de maior valor (Refinanciamento Imobiliário e CDB) vêm da política Thompson Sampling já treinada e coincidem com o braço de maior valor esperado entre os elegíveis; os casos da Conta Digital e do Cartão Premium vêm de impressões de exploração (Thompson Sampling) ou da regra fixa (baseline), já que essas duas ofertas raramente são a escolha ótima quando o cliente também é elegível para uma oferta mais valiosa. O detalhamento completo está no notebook [`src/eda/Etapa3_Etapa4_Bandit_Analysis.ipynb`](src/eda/Etapa3_Etapa4_Bandit_Analysis.ipynb).

| Oferta | Cliente (`client_id`) | Idade | Financ. habitacional | Política (origem) | Recompensa | Bate com o ótimo teórico? |
|---|---|---|---|---|---|---|
| Conta Digital Padrão | 269 | 40 | sim | BaselineFixedPolicy | R$ 10,00 | Não |
| Conta Digital Padrão | 20076 | 32 | não | BaselineFixedPolicy | R$ 10,00 | Não |
| Conta Digital Padrão | 15029 | 55 | não | BaselineFixedPolicy | R$ 10,00 | Não |
| Conta Digital Padrão | 11094 | 48 | sim | BaselineFixedPolicy | R$ 10,00 | Não |
| Conta Digital Padrão | 41173 | 32 | não | BaselineFixedPolicy | R$ 10,00 | Não |
| Cartão de Crédito Premium | 17979 | 54 | não | ThompsonSamplingPolicy | R$ 150,00 | Não |
| Cartão de Crédito Premium | 9469 | 56 | não | ThompsonSamplingPolicy | R$ 150,00 | Não |
| Cartão de Crédito Premium | 23215 | 38 | não | ThompsonSamplingPolicy | R$ 150,00 | Não |
| Cartão de Crédito Premium | 7802 | 39 | não | ThompsonSamplingPolicy | R$ 150,00 | Não |
| Cartão de Crédito Premium | 24670 | 52 | não | ThompsonSamplingPolicy | R$ 150,00 | Não |
| Refinanciamento Imobiliário | 7265 | 39 | sim | ThompsonSamplingPolicy | R$ 300,00 | Sim |
| Refinanciamento Imobiliário | 8886 | 44 | sim | ThompsonSamplingPolicy | R$ 300,00 | Sim |
| Refinanciamento Imobiliário | 32528 | 37 | sim | ThompsonSamplingPolicy | R$ 300,00 | Sim |
| Refinanciamento Imobiliário | 26830 | 41 | sim | ThompsonSamplingPolicy | R$ 300,00 | Sim |
| Refinanciamento Imobiliário | 17264 | 28 | sim | ThompsonSamplingPolicy | R$ 300,00 | Sim |
| Depósito a Prazo (CDB) | 28239 | 32 | não | ThompsonSamplingPolicy | R$ 80,00 | Sim |
| Depósito a Prazo (CDB) | 40299 | 37 | não | ThompsonSamplingPolicy | R$ 80,00 | Sim |
| Depósito a Prazo (CDB) | 22910 | 34 | não | ThompsonSamplingPolicy | R$ 80,00 | Sim |
| Depósito a Prazo (CDB) | 39545 | 30 | não | ThompsonSamplingPolicy | R$ 80,00 | Sim |
| Depósito a Prazo (CDB) | 20344 | 43 | não | ThompsonSamplingPolicy | R$ 80,00 | Sim |

## Etapa 6 — Arquitetura-alvo em Nuvem (GCP)

Para colocar este projeto em produção, o grupo utilizaria a Google Cloud Platform (GCP). Os dados (base de clientes, catálogo de ofertas, Golden Set e o índice vetorial do RAG) ficariam no **Cloud Storage**, com o **BigQuery** guardando as métricas dos experimentos para consultas analíticas. A geração de eventos sintéticos e a simulação das políticas bandit rodariam como jobs agendados via **Cloud Scheduler** disparando **Cloud Run Jobs**, enquanto a API que recomenda ofertas em tempo real (a política Thompson Sampling treinada) seria exposta como um serviço **Cloud Run**, escalando automaticamente conforme a demanda sem precisar gerenciar servidores.

O assistente de RAG/LLM (hoje rodando localmente com Gemini e Groq como fallback) seria outro serviço **Cloud Run**, com as chaves de API guardadas no **Secret Manager** em vez de arquivos `.env`. As imagens de container seriam construídas e versionadas com **Cloud Build** e **Artifact Registry**, e o **Cloud Logging**/**Cloud Monitoring** cuidariam da observabilidade (erros, latência, uso de cada serviço), com o **IAM** controlando quem acessa cada recurso. A criação de diagramas de arquitetura é opcional e não foi incluída nesta etapa.

## Licença

Este repositório não informa uma licença explícita no momento.
