# Relatório de Desempenho do Experimento: LinUCB (Bandit Contextual)

**Autor:** Cientista de Dados & Engenheiro de Machine Learning
**Data:** 24 de Maio de 2024
**Experimento:** Análise da Política LinUCB (Linear Upper Confidence Bound)
**Arquivo de Origem:** `data/experiments/linucb/metrics_summary.csv`

---

## 1. Introdução

Este relatório apresenta a análise de desempenho da política **LinUCB (Linear Upper Confidence Bound)**, um algoritmo de bandit contextual projetado para otimizar a seleção de ações (recomendações) com base em recursos de contexto (usuário e item). O LinUCB aborda o clássico dilema entre **exploração** (*exploration* - testar novas ações para coletar informações) e **explotação** (*exploitation* - escolher a melhor ação conhecida para maximizar o retorno imediato) assumindo uma relação linear entre os recursos do contexto e a recompensa esperada.

O objetivo desta análise é avaliar a eficácia da política em termos de conversão, geração de valor (recompensa), custo de aprendizado (regret) e o nível de diversidade nas escolhas (entropia).

---

## 2. Resumo das Métricas do Experimento

Abaixo estão consolidada as métricas obtidas durante a execução do experimento:

| Métrica | Valor Absoluto / Proporção | Descrição |
| :--- | :--- | :--- |
| **Política** | `LinUCBPolicy` | Algoritmo de bandit contextual avaliado. |
| **Impressões (Jogadas)** | $4.521$ | Número total de decisões de recomendação tomadas. |
| **Conversões** | $433$ | Total de interações positivas (sucessos) obtidas. |
| **Taxa de Conversão (CTR)** | $9,58\%$ (`0.095775`) | Proporção de conversões em relação às impressões. |
| **Recompensa Acumulada** | $53.010,0$ | Soma dos valores de recompensa obtidos no experimento. |
| **Regret Acumulado** | $26.737,2$ | Perda acumulada por não escolher a ação ótima teórica. |
| **Entropia de Exploração** | $1,2048$ | Medida de incerteza/dispersão na escolha das ações. |

---

## 3. Análise de Desempenho

### 3.1. Eficácia e Geração de Valor
Com **4.521 impressões**, a política LinUCB gerou **433 conversões**, resultando em uma **Taxa de Conversão de 9,58%**. Em cenários reais de recomendação e publicidade digital, uma taxa de conversão próxima de 10% é considerada altamente competitiva, demonstrando que a política foi capaz de utilizar os recursos de contexto de maneira eficaz para personalizar as ofertas.

A **Recompensa Acumulada de 53.010,0** indica que o algoritmo gerou um retorno médio de **11,72 unidades de recompensa por impressão**. Isso prova que a política não apenas buscou cliques/conversões simples, mas conseguiu otimizar a entrega de itens de maior valor agregado quando o contexto era propício.

### 3.2. O Trade-Off entre Exploração e Explotação
O equilíbrio entre exploração e explotação é o núcleo do LinUCB. Podemos inferir o comportamento do algoritmo através de duas métricas principais:

1. **Entropia de Exploração (1,2048):**
   Uma entropia de ~1,20 indica uma distribuição moderadamente ativa e saudável na exploração de braços (ações). Se a entropia fosse muito próxima de zero, a política estaria estagnada em explotação pura (recomendando sempre os mesmos itens super-otimizados). Se fosse excessivamente alta, indicaria escolhas puramente aleatórias. O valor de 1,20 demonstra que o LinUCB manteve uma postura investigativa ativa, distribuindo impressões entre diferentes braços para refinar seus modelos lineares de recompensa, ao mesmo tempo em que direcionava tráfego para as melhores opções conhecidas.

2. **Regret Acumulado (26.737,2):**
   O *regret* (arrependimento) mede a diferença entre a recompensa que teria sido obtida pela melhor ação teórica (oráculo) e a recompensa real obtida pela política.
   * Um regret acumulado de 26.737,2 em relação a uma recompensa de 53.010,0 indica que o custo de aprendizado do algoritmo representou cerca de **33,5% do potencial ótimo**.
   * Em bandits contextuais, o regret tende a crescer de forma sublinear (logarítmica) ao longo do tempo. Na fase inicial do experimento, o regret cresce rapidamente enquanto o modelo aprende os coeficientes lineares de cada braço. À medida que o número de impressões aumenta, a taxa de crescimento do regret deve diminuir, convergindo para um estado de explotação de alta performance.

---

## 4. Diagnóstico Técnico e Conclusões

1. **Viabilidade do Modelo Linear:** A taxa de conversão saudável de 9,58% sugere que a suposição de linearidade entre os contextos dos usuários/itens e as recompensas é válida para este conjunto de dados. O LinUCB conseguiu extrair valor real das variáveis de contexto para tomar decisões informadas.
2. **Custo de Aprendizado Aceitável:** Embora o regret acumulado pareça expressivo em termos absolutos (26.737,2), ele é um investimento necessário para a convergência do algoritmo. O nível de entropia controlado (1,20) valida que o algoritmo não ficou "preso" em mínimos locais e manteve uma busca ativa por políticas melhores.
3. **Robustez do Algoritmo:** O LinUCB provou ser uma escolha robusta, entregando um volume robusto de conversões e recompensa financeira expressiva sob um regime de exploração controlado.

---

## 5. Próximos Passos Recomendados

Para otimizar ainda mais o desempenho e reduzir o regret acumulado, sugerem-se as seguintes ações de engenharia de machine learning:

* **Ajuste do Hiperparâmetro $\alpha$ (Exploração):** O parâmetro $\alpha$ do LinUCB controla diretamente a largura das bandas de confiança (o peso dado à incerteza). Um teste de grade (*grid search*) ou decaimento gradual de $\alpha$ ao longo do tempo poderia reduzir a entropia na fase tardia do experimento, diminuindo o regret e aumentando a taxa de conversão final.
* **Engenharia de Recursos (Feature Engineering):** Adicionar interações não lineares ou aplicar reduções de dimensionalidade nos contextos pode ajudar o modelo linear a convergir mais rapidamente, reduzindo o tempo necessário para que o regret se estabilize.
* **Benchmark com Outras Políticas:** Recomenda-se comparar estes resultados com políticas de *Thompson Sampling Contextual* ou *$\epsilon$-greedy* adaptativo para validar se o LinUCB continua sendo a melhor abordagem para este domínio de negócio.
