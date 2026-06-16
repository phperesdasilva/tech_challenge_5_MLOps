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

- `OUT_DIR`: diretório de saída do experimento.
- `SEED`: semente aleatória do experimento.
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

## Licença

Este repositório não informa uma licença explícita no momento.
