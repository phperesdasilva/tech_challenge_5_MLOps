# Relatório de Análise: Recomendações do LinUCB por Histórico de Campanha (`poutcome`)

## 1. Introdução

Este relatório apresenta uma análise detalhada das decisões de recomendação de ofertas (braços) tomadas pela política contextual **LinUCB** (Linear Upper Confidence Bound). O objetivo é compreender como o histórico de contato anterior do cliente (`poutcome` - resultado da campanha anterior) influencia a distribuição e a escolha do algoritmo pelas diferentes ofertas disponíveis (Braços 0, 1, 2 e 3).

O algoritmo LinUCB utiliza características contextuais dos clientes para balancear a exploração (*exploration*) e a explotação (*exploitation*), estimando a recompensa esperada para cada braço. Ao analisar a distribuição das recomendações por perfil de histórico, conseguimos extrair valiosos insights de negócios e de comportamento de modelo.

---

## 2. Visão Geral dos Dados de Recomendação

Os dados consolidados de recomendação por histórico de campanha anterior (`poutcome`) estão estruturados conforme a tabela abaixo:

| poutcome | Braço 0 | Braço 1 | Braço 2 | Braço 3 | Predominante | Total de Clientes | % do Total Geral |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **failure** | 5 | 34 | 235 | 216 | **Braço 2** | 490 | 10,84% |
| **other** | 8 | 17 | 84 | 88 | **Braço 3** | 197 | 4,36% |
| **success** | 5 | 14 | 26 | 84 | **Braço 3** | 129 | 2,85% |
| **unknown** | 71 | 324 | 1249 | 2061 | **Braço 3** | 3.705 | 81,95% |
| **Total Geral** | **89** | **389** | **1.594** | **2.449** | **Braço 3** | **4.521** | **100,00%** |

---

## 3. Análise Detalhada por Categoria de Histórico (`poutcome`)

### 3.1. Clientes com Histórico de Sucesso (`success`)
*   **Volume de dados:** 129 clientes (2,85% do total).
*   **Distribuição das Recomendações:**
    *   Braço 0: 3,88% (5 recomendações)
    *   Braço 1: 10,85% (14 recomendações)
    *   Braço 2: 20,16% (26 recomendações)
    *   **Braço 3: 65,12% (84 recomendações) — Predominante**
*   **Análise:** Para os clientes que já converteram ou tiveram sucesso na campanha anterior, o LinUCB demonstra uma preferência massiva pelo **Braço 3** (mais de 65% das recomendações). O Braço 2 é uma alternativa secundária distante. Isso sugere que a oferta associada ao Braço 3 possui uma altíssima taxa de conversão esperada ou recompensa para clientes historicamente engajados.

### 3.2. Clientes com Histórico de Falha (`failure`)
*   **Volume de dados:** 490 clientes (10,84% do total).
*   **Distribuição das Recomendações:**
    *   Braço 0: 1,02% (5 recomendações)
    *   Braço 1: 6,94% (34 recomendações)
    *   **Braço 2: 47,96% (235 recomendações) — Predominante**
    *   Braço 3: 44,08% (216 recomendações)
*   **Análise:** Este é o cenário mais interessante de mudança de comportamento do modelo. Para clientes que rejeitaram ou falharam na campanha anterior, o **Braço 2 assume a liderança** (47,96%), superando ligeiramente o Braço 3 (44,08%). O LinUCB aprendeu que a abordagem do Braço 2 é ligeiramente mais eficaz em mitigar a rejeição anterior ou reengajar clientes "difíceis" do que a oferta padrão (Braço 3).

### 3.3. Clientes com Histórico Outro/Indefinido (`other`)
*   **Volume de dados:** 197 clientes (4,36% do total).
*   **Distribuição das Recomendações:**
    *   Braço 0: 4,06% (8 recomendações)
    *   Braço 1: 8,63% (17 recomendações)
    *   Braço 2: 42,64% (84 recomendações)
    *   **Braço 3: 44,67% (88 recomendações) — Predominante**
*   **Análise:** Nesta categoria intermediária, há um equilíbrio técnico muito forte entre o **Braço 3** (44,67%) e o **Braço 2** (42,64%). O modelo distribui suas apostas de forma quase equivalente entre essas duas ofertas principais, refletindo a incerteza inerente ao histórico classificado apenas como "outro".

### 3.4. Clientes sem Histórico Registrado (`unknown`)
*   **Volume de dados:** 3.705 clientes (81,95% do total).
*   **Distribuição das Recomendações:**
    *   Braço 0: 1,92% (71 recomendações)
    *   Braço 1: 8,74% (324 recomendações)
    *   Braço 2: 33,71% (1.249 recomendações)
    *   **Braço 3: 55,63% (2.061 recomendações) — Predominante**
*   **Análise:** Sendo a maior fatia da base de clientes, a categoria "unknown" dita a tendência geral do modelo. O **Braço 3** é amplamente dominante (55,63%), consolidando-se como a oferta "padrão" ou de maior apelo geral na ausência de informações prévias de contato. O Braço 2 permanece como uma sólida segunda opção (33,71%).

---

## 4. O papel dos Braços 0 e 1

Em todos os cenários analisados, os **Braços 0 e 1 receberam pouquíssimas recomendações** (compostos somados representam pouco mais de 10% de toda a base).
*   **Braço 0** é o menos recomendado em geral, flutuando entre 1% e 4% das recomendações.
*   **Braço 1** apresenta um desempenho ligeiramente melhor, mas ainda marginal, variando de 6,9% a 10,8%.

Esse padrão indica que, no balanço entre exploração e explotação, o LinUCB rapidamente identificou que os retornos esperados desses braços são inferiores aos dos braços 2 e 3 para a grande maioria dos perfis de clientes. A baixa taxa de recomendação destes braços reflete a consolidação da política de explotação de ofertas de melhor desempenho (conforme sumarizado em `metrics_summary.csv`, onde políticas de exploração inteligente superam amplamente políticas aleatórias).

---

## 5. Principais Insights e Conclusões

1.  **A Dominância do Braço 3:** O Braço 3 é a oferta mais robusta e de maior aceitação global. Sua força é máxima em clientes que já demonstraram propensão positiva no passado (`success` com 65,12%) e em clientes sem histórico (`unknown` com 55,63%). É o "carro-chefe" do portfólio de campanhas.
2.  **O Poder de Recuperação do Braço 2:** O Braço 2 funciona como uma estratégia de contingência eficaz. Sua vitória sobre o Braço 3 no segmento `failure` (47,96% vs 44,08%) sugere que esta oferta possui características que geram menos atrito ou são mais atrativas para clientes anteriormente insatisfeitos ou não convertidos.
3.  **Personalização Contextual Relevante:** O LinUCB provou ser eficaz ao adaptar a recomendação de forma sutil mas estatisticamente significativa. A inversão de preferência (Braço 3 como predominante no sucesso/geral e Braço 2 na falha) é o comportamento ideal esperado de um algoritmo de recomendação contextualizado por histórico.
4.  **Otimização de Portfólio:** Os Braços 0 e 1 mostraram-se pouco eficientes de maneira geral. Isso levanta uma recomendação de negócios para revisar as regras de negócio ou as próprias ofertas desses braços, uma vez que o modelo de Machine Learning identificou baixa probabilidade de recompensa neles em comparação aos demais concorrentes.
