| Campo | Conteúdo |
|---|---|
| Nome | Política Thompson Sampling — ofertas bancárias |
| Tipo | Multi-armed bandit (Bernoulli por braço) |
| Inputs | Cliente (age, housing, balance); catálogo de ofertas |
| Output | arm_id selecionado |
| Baseline | BaselineFixedPolicy (arm 0) |
| Métricas | Reward, regret, CR, entropia de exploração |
| Limitações | Sem contextualização; priors sintéticos |
| Cold-start | Beta(1,1) |
| Delayed rewards | Fila com conversion_timestamp |

## Pré-requisito: clean_bank.parquet

O script referencia `data/kaggle/processed/clean_bank.parquet`. Se ainda não existe, rode o pipeline da Etapa 1/2 (EDA) ou crie um parquet mínimo com colunas `age`, `balance`, `housing` a partir do `bank-full.csv`.