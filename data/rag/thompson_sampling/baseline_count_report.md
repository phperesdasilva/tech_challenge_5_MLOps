# Relatório de Análise de Execução: BaselineFixedPolicy

## 1. Introdução
Este relatório apresenta a análise de distribuição de execuções (alocação de braços) para a política **BaselineFixedPolicy**, utilizada como grupo de controle no experimento de Multi-Armed Bandits (MAB) com Thompson Sampling.

O objetivo de uma política estática (ou baseline fixa) é estabelecer uma linha de base de desempenho para comparar a eficácia de algoritmos adaptativos, como o Thompson Sampling. Ao fixar a escolha em um único braço padrão, podemos quantificar o ganho real (*uplift*) gerado pela inteligência e capacidade de aprendizado do algoritmo MAB.

## 2. Distribuição de Execução dos Braços

Com base nos dados coletados no arquivo `arm_counts_BaselineFixedPolicy.csv`, a distribuição de jogadas (*pulls*) por braço está detalhada na tabela abaixo:

| Política | ID do Braço | Contagem de Execuções | Proporção (%) |
| :--- | :---: | :---: | :---: |
| **BaselineFixedPolicy** | 0 | 4.521 | 100,00% |
| **Total** | - | **4.521** | **100,00%** |

### Observações Chave:
* **Explotação Exclusiva:** Como esperado de uma política de controle fixa, 100% das 4.521 interações foram direcionadas ao **Braço 0** (`arm_id: 0`).
* **Ausência de Exploração (Exploration):** Não houve qualquer tentativa de explorar outros braços alternativos disponíveis no experimento. O sistema operou de forma determinística e estática.

## 3. Implicações Teóricas e Práticas da Política

A escolha de alocar todas as execuções ao Braço 0 traz implicações significativas para a avaliação do experimento:

1. **Viés de Amostragem:** Por não explorar outros braços, esta política assume a premissa de que o Braço 0 é a escolha padrão histórica ou a opção de menor risco. Ela não se adapta a mudanças no comportamento do usuário ou a variações sazonais.
2. **Custo de Oportunidade:** Caso o Braço 0 não seja o braço com a maior taxa de recompensa (conversão/clique), a política incorre em um custo de oportunidade contínuo, representado pelo *regret* acumulado em comparação com uma política ótima ou com o Thompson Sampling.
3. **Papel como Controle:** Essa distribuição estática simula o comportamento de um grupo de controle em um teste A/B tradicional, onde os usuários do grupo são expostos a uma única variante pré-definida.

## 4. Contextualização com Métricas de Desempenho

Utilizando os dados agregados de `metrics_summary.csv` como referência analítica, o desempenho consolidado desta política serve como o "ponto zero" para o cálculo das métricas de sucesso do Thompson Sampling:

* **Taxa de Conversão/Recompensa Média:** A taxa de conversão obtida pelo Braço 0 sob esta política reflete o comportamento orgânico da população sem otimização dinâmica.
* **Regret Acumulado:** O regret desta política cresce de forma linear ao longo do tempo caso exista um braço com desempenho superior disponível no experimento, servindo como o limite inferior de desempenho aceitável para o experimento.

## 5. Conclusão

A política `BaselineFixedPolicy` cumpriu rigorosamente o seu papel operacional, executando o **Braço 0 exatamente 4.521 vezes**.

Este volume de dados é estatisticamente robusto para determinar com alta precisão o desempenho médio do cenário padrão (controle). A comparação direta desta volumetria e de sua respectiva taxa de conversão contra as métricas do Thompson Sampling permitirá validar se o algoritmo adaptativo foi capaz de identificar braços superiores e realizar a transição de exploração para explotação de forma eficiente, mitigando o *regret* acumulado e maximizando o retorno sobre o experimento.
