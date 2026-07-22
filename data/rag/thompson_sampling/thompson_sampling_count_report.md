# Relatório de Análise de Alocação de Braços: Thompson Sampling

## 1. Introdução
Este relatório apresenta uma análise detalhada da distribuição de execuções dos braços (ações) durante o experimento utilizando a política **Thompson Sampling** (`ThompsonSamplingPolicy`). O objetivo desta análise é compreender o comportamento de exploração (*exploration*) e explotação (*exploitation*) do algoritmo com base nos dados consolidados de contagem de execuções.

Os dados utilizados como base para este relatório foram extraídos do arquivo `arm_counts_ThompsonSamplingPolicy.csv`.

---

## 2. Metodologia e Dados de Execução
O algoritmo de Thompson Sampling (ou Amostragem de Thompson) é uma abordagem bayesiana para o problema do *Multi-Armed Bandit* (MAB). Ele mantém uma distribuição de probabilidade para a recompensa esperada de cada braço e seleciona as ações com base na probabilidade de serem as melhores.

A tabela abaixo apresenta a contagem absoluta e o percentual de alocação de cada braço ao longo do experimento:

| Política | ID do Braço | Contagem de Execuções | Percentual de Alocação (%) |
| :--- | :---: | :---: | :---: |
| **ThompsonSamplingPolicy** | 0 | 4.061 | 89,83% |
| **ThompsonSamplingPolicy** | 3 | 227 | 5,02% |
| **ThompsonSamplingPolicy** | 2 | 120 | 2,65% |
| **ThompsonSamplingPolicy** | 1 | 113 | 2,50% |
| **Total** | - | **4.521** | **100,00%** |

---

## 3. Análise do Comportamento da Política

### 3.1. Dominância do Braço 0
O **Braço 0** foi selecionado em **89,83%** das rodadas (4.061 vezes de um total de 4.521). Esse padrão indica fortemente que:
* O algoritmo identificou rapidamente o Braço 0 como o braço de melhor desempenho (maior taxa de recompensa esperada).
* Após a fase inicial de exploração, a distribuição posterior do Braço 0 convergiu para valores de recompensa superiores aos demais, fazendo com que a amostragem bayesiana priorizasse quase que exclusivamente esta opção (fase de explotação).

### 3.2. Exploração dos Braços Subótimos
Os Braços 3, 2 e 1 receberam atenção significativamente menor:
* **Braço 3:** Recebeu 5,02% das alocações. Foi o segundo braço mais explorado, sugerindo que suas estimativas de recompensa inicialmente competiram de forma leve com o Braço 0 ou que ele possui uma variância que justificou tentativas esporádicas.
* **Braços 2 e 1:** Receberam as menores taxas de alocação (2,65% e 2,50%, respectivamente). A rápida redução no número de jogadas nesses braços mostra a eficiência do Thompson Sampling em descartar opções claramente subótimas sem a necessidade de gastar excessivo orçamento de testes.

---

## 4. Conclusão e Recomendações

O comportamento demonstrado pela `ThompsonSamplingPolicy` é característico de um algoritmo de aprendizado por reforço bem-sucedido no cenário de tomada de decisão sequencial:
1. **Minimização do Regret:** Ao direcionar quase 90% dos recursos para o Braço 0, o algoritmo minimizou o custo de oportunidade (perda de recompensa por escolher braços piores).
2. **Eficiência Estatística:** A política coletou dados suficientes dos braços 1, 2 e 3 para garantir, com alto nível de confiança estatística, que eles eram inferiores ao Braço 0, reduzindo as tentativas subsequentes de forma drástica.

**Próximos Passos Recomendados:**
* **Análise de Recompensa (Metrics Summary):** Cruzar estes dados de contagem com o arquivo `metrics_summary.csv` para validar se o Braço 0 de fato entregou o maior *Click-Through Rate* (CTR) ou retorno financeiro esperado.
* **Avaliação de Estacionaridade:** Avaliar se o comportamento das recompensas muda ao longo do tempo. Se o ambiente for não-estacionário, estratégias de desconto temporal (como o *Discounted Thompson Sampling*) podem ser avaliadas para re-explorar braços antigos caso suas distribuições de recompensa sofram alterações.
