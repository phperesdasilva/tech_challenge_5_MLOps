# Tech Challenge 5 - MLOps

Projeto de MLOps voltado à análise de dados de marketing bancário e à simulação de políticas de recomendação com multi-armed bandits. O repositório reúne a preparação dos dados, a geração de eventos sintéticos, o experimento da etapa 3 e os artefatos de documentação do modelo.

## Visão Geral

O fluxo principal do projeto é:

1. Explorar e preparar os dados do banco de marketing.
2. Construir um catálogo de ofertas com regras de elegibilidade e recompensa sintética.
3. Gerar eventos sintéticos de impressão e conversão com atraso.
4. Simular políticas bandit, como baseline fixa, Thompson Sampling e UCB1.
5. Salvar métricas e históricos para análise posterior.

## Estrutura do Repositório

- `src/EDA.ipynb`: notebook de exploração e análise inicial dos dados.
- `src/run_etapa3_experiment.py`: script principal que executa o experimento da etapa 3.
- `src/bandit/`: implementação do catálogo, ambiente, métricas, políticas e simulador.
- `src/event_generator/`: geração de eventos sintéticos e recompensas atrasadas.
- `data/`: dados brutos, processados, conjuntos sintéticos e resultados de experimentos.
- `docs/model-card.md`: resumo do modelo e de suas limitações.
- `tests/`: testes automatizados.

## Requisitos

- Python `>=3.12.3`
- Dependências principais: `numpy`, `pandas`, `pyarrow`, `python-dotenv`
- Dependências opcionais:
	- Desenvolvimento: `pytest`, `pytest-cov`, `pre-commit`, `black`, `isort`
	- MLOps: `kaggle`, `mlflow`

## Instalação

Crie e ative o ambiente virtual, depois instale o projeto em modo editável:

```bash
pip install -e .
```

Para instalar extras de desenvolvimento:

```bash
pip install -e .[dev]
```

Para instalar extras de MLOps:

```bash
pip install -e .[mlops]
```

Para instalar tudo junto:

```bash
pip install -e .[dev,mlops]
```

## Dados

O projeto espera, por padrão, os seguintes artefatos:

- `data/kaggle/processed/clean_bank.parquet`
- `data/kaggle/synthetic_enrichment/offer_catalog.json`
- `data/kaggle/synthetic_enrichment/offer_events.csv`
- `data/kaggle/synthetic_enrichment/delayed_rewards.csv`

O notebook de EDA e o script de geração sintética partem da base do Bank Marketing disponível em `data/kaggle/raw/bank-marketing-data-set/`.

## Como Executar

### 1. Análise exploratória

Abra o notebook `src/EDA.ipynb` no VS Code ou no Jupyter para revisar a preparação inicial dos dados.

### 2. Gerar eventos sintéticos

O pacote expõe um comando de console para gerar eventos e recompensas atrasadas:

```bash
generate-events
```

Esse comando produz arquivos em `data/kaggle/synthetic_enrichment/`.

### 3. Rodar o experimento da etapa 3

Execute o script principal a partir da raiz do projeto:

```bash
python src/run_etapa3_experiment.py
```

Os resultados são salvos por padrão em `data/experiments/etapa3/`.

## Saídas do Experimento

O experimento gera três artefatos principais:

- `metrics_summary.csv`: resumo agregado das métricas por política.
- `metrics_timeseries.parquet`: série temporal das métricas registradas.
- `arm_counts_<policy>.csv`: contagem de seleções por braço e política.

## Variáveis de Ambiente

As seguintes variáveis podem ser definidas via `.env`:

- `TS_OUT_DIR`: diretório de saída do experimento.
- `SEED`: semente aleatória do experimento Thompson Sampling.
- `DEFAULT_BANK_PATH`: caminho para `clean_bank.parquet`.
- `DEFAULT_CATALOG_PATH`: caminho para o catálogo de ofertas.
- `BASE_DATE`: data base da simulação.
- `DELAY_SCALE_DAYS`: escala do atraso das conversões.
- `THOMPSON_ALPHA0` e `THOMPSON_BETA0`: priors da política Thompson Sampling.
- `UCB1_EXPLORATION_BONUS`: fator de exploração da política UCB1.

## Políticas Implementadas

- `BaselineFixedPolicy`: sempre tenta o braço `0` quando elegível.
- `ThompsonSamplingPolicy`: seleciona braços por amostragem Beta por recompensa binária.
- `UCB1Policy`: usa o índice UCB para balancear exploração e exploração.

## Limitações Conhecidas

- A abordagem não é contextual; os clientes não influenciam diretamente o score além da elegibilidade.
- Os priors e recompensas sintéticas dependem de hipóteses simplificadas.
- O pipeline assume a existência do parquet processado `clean_bank.parquet`.

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
