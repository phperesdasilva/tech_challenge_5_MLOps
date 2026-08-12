# Relatório de Análise: Distribuição de Alocação da Política BaselineFixedPolicy

## 1. Introdução
Este relatório apresenta a análise da distribuição de execuções de braços para a política **BaselineFixedPolicy**, utilizada como grupo de controle (baseline) nos experimentos de Multi-Armed Bandit (MAB) comparados com o algoritmo Thompson Sampling.

O objetivo do estabelecimento de uma política de controle fixa é fornecer um benchmark de desempenho estático para avaliar o ganho acumulado, a taxa de conversão e o arrependimento (regret) das políticas adaptativas de Thompson Sampling.

---

## 2. Metodologia e Dados de Execução
Os dados de execução foram extraídos do arquivo `arm_counts_BaselineFixedPolicy.csv`, que registra o volume de alocações para cada braço sob esta política.

### Tabela 1: Distribuição de Alocações por Braço
| Política | ID do Braço | Contagem de Execuções (Counts) | Proporção de Alocação (%) |
| :--- | :---: | :---: | :---: |
| **BaselineFixedPolicy** | 0 | 4.521 | 100,00% |
| **Total** | - | **4.521** | **100,00%** |

---

## 3. Análise de Comportamento da Política
A partir dos dados observados, nota-se que o braço **0** foi executado exatamente **4.521 vezes**, representando **100%** de todas as interações do experimento para esta política.

Este comportamento é característico e esperado para a `BaselineFixedPolicy`. Como uma política estática e não adaptativa, ela não realiza exploração (*exploration*) nem aproveita aprendizados para explotação (*exploitation*). Ela simplesmente direciona todo o tráfego/fluxo para um único braço pré-definido (neste caso, o Braço 0).

---

## 4. Relação com o Resumo de Métricas (metrics_summary.csv)
Ao correlacionar estes dados com o arquivo de referência geral de métricas (`metrics_summary.csv`), podemos extrair conclusões valiosas sobre a performance do Baseline:

1. **Taxa de Conversão Constante:** A taxa de conversão (CTR) observada para esta política reflete puramente a taxa de conversão basal do Braço 0.
2. **Ausência de Aprendizado:** Diferente das variantes de Thompson Sampling (que dinamicamente migram o tráfego para os braços de melhor performance ao longo do tempo), a `BaselineFixedPolicy` mantém sua performance linear e constante do início ao fim das 4.521 rodadas.
3. **Cálculo de Regret:** Esta execução de 4.521 interações serve como o ponto de ancoragem para o cálculo de *Regret* Acumulado das outras políticas. Se o Braço 0 for subótimo em comparação com outros braços disponíveis no experimento, a curva de regret desta política crescerá de forma linear e acentuada.

---

## 5. Conclusão
A política `BaselineFixedPolicy` cumpriu rigorosamente o seu papel metodológico de controle, concentrando 100% de suas 4.521 execuções no Braço 0.

A estabilidade desta política é fundamental para isolar variáveis externas e garantir que quaisquer melhorias de performance observadas nas políticas baseadas em Thompson Sampling sejam estatisticamente atribuíveis à inteligência do algoritmo de alocação dinâmica, e não a flutuações aleatórias do ambiente do experimento.
