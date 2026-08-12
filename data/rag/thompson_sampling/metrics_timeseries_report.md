# Relatório de Análise Comparativa: Thompson Sampling vs. Baseline Fixed Policy

**Autor:** Cientista de Dados / Engenheiro de Machine Learning
**Data da Análise:** October 2023
**Experimento:** Otimização de Multi-Armed Bandits via Thompson Sampling

---

## 1. Sumário Executivo

Este relatório apresenta uma análise detalhada dos resultados obtidos no experimento comparativo entre o algoritmo de **Thompson Sampling** (uma política de tomada de decisão sequencial sob incerteza) e a **Baseline Fixed Policy** (uma política fixa de controle).

Os dados analisados cobrem um horizonte de **4.521 passos (steps)**. O principal insight gerado por este experimento é que **otimizar apenas a taxa de conversão pode ser prejudicial para o retorno financeiro ou valor total de conversão**. O modelo de Thompson Sampling obteve um desempenho financeiro drasticamente superior, acumulando **16.770,0 de recompensa total** contra **6.920,0 da Baseline** (um aumento de **142,34%**), além de reduzir o arrependimento acumulado (*cumulative regret*) em **13,53%**, mesmo registrando uma taxa de conversão ligeiramente menor (13,96% vs. 15,31%).

---

## 2. Metodologia

O experimento foi estruturado como um problema de *Multi-Armed Bandits* (MAB):
*   **Baseline Fixed Policy:** Representa uma política estática onde as decisões são tomadas com base em regras fixas pré-determinadas (geralmente explorando uma única opção considerada padrão ou segura).
*   **Thompson Sampling Policy:** Uma abordagem bayesiana para o equilíbrio entre exploração e aproveitamento (*exploration vs. exploitation*). O algoritmo modela a incerteza de cada braço (ação) usando distribuições de probabilidade (tipicamente distribuições Beta para recompensas binárias ou conjugadas apropriadas para recompensas contínuas) e toma decisões amostrando dessas distribuições.

---

## 3. Comparação de Métricas Finais

Ao final dos 4.521 passos do experimento, as métricas consolidadas apresentaram o seguinte cenário:

| Métrica | Baseline Fixed Policy | Thompson Sampling Policy | Diferença Absoluta | Variação (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Recompensa Acumulada** | 6.920,0 | 16.770,0 | +9.850,0 | **+142,34%** |
| **Regret Acumulado** | 72.827,2 | 62.977,2 | -9.850,0 | **-13,53%** |
| **Taxa de Conversão Final** | 15,31% | 13,96% | -1,35 pp | **-8,82%** |

---

## 4. Análise de Evolução Temporal e Progressão

### 4.1. Recompensa Acumulada (*Cumulative Reward*)
A progressão da recompensa acumulada revela uma divergência clara e precoce entre as duas políticas:
*   **Fase Inicial (Passos 1 a 100):** O Thompson Sampling rapidamente se destaca. Logo no passo 17, o modelo de Thompson Sampling já acumulava uma recompensa de 900,0, enquanto a Baseline estava em apenas 30,0. Esse comportamento inicial agressivo reflete a capacidade do Thompson Sampling de identificar braços de altíssimo valor logo nas primeiras interações.
*   **Fase de Estabilização:** A curva de recompensa da Baseline cresce de forma linear e lenta ao longo de todo o experimento. Em contrapartida, o Thompson Sampling mantém uma taxa de inclinação consistentemente superior, acumulando valor de forma muito mais eficiente.

### 4.2. Arrependimento Acumulado (*Cumulative Regret*)
O *regret* mede a perda acumulada por não escolher a melhor ação ótima teórica em cada passo:
*   **Evolução da Baseline:** O arrependimento da Baseline cresce de forma estritamente linear, alcançando seu pico de 72.827,2 no último passo. Isso indica que a política fixa continuou a tomar decisões subótimas de maneira consistente, sem qualquer capacidade de adaptação.
*   **Evolução do Thompson Sampling:** O arrependimento acumulado do Thompson Sampling inicia com valores negativos em determinados momentos do início do experimento (ex: -578,4 no passo 17). Isso ocorre devido à formulação do cálculo de arrependimento em relação a um valor esperado de referência, onde o Thompson Sampling conseguiu selecionar braços com retornos excepcionalmente altos e acima da média esperada. Ao longo do tempo, o crescimento do arrependimento do modelo desacelera significativamente em comparação com a Baseline, terminando 13,53% menor.

### 4.3. Taxa de Conversão (*Conversion Rate*)
Um dos pontos mais contra-intuitivos do experimento reside na taxa de conversão:
*   A **Baseline** manteve uma taxa de conversão bastante estável, flutuando entre 15,0% e 16,0% na maior parte do tempo, encerrando em **15,31%**.
*   O **Thompson Sampling** iniciou com conversões flutuantes e estabilizou em uma trajetória descendente suave, encerrando em **13,96%**.

---

## 5. Principais Insights e Discussão Business-Oriented

### Insight 1: O Paradoxo da Taxa de Conversão vs. Recompensa Total
O resultado mais valioso deste experimento é a demonstração prática de que **taxa de conversão não é sinônimo de receita ou valor de negócio**.
*   A Baseline converteu mais frequentemente (15,31%), porém suas conversões geraram recompensas individuais muito baixas (provavelmente focando em itens baratos ou ofertas de baixo valor).
*   O Thompson Sampling converteu menos frequentemente (13,96%), mas aprendeu de forma inteligente a priorizar braços que entregam uma **recompensa significativamente maior por conversão**. Na prática, o algoritmo preferiu falhar ligeiramente mais em converter, desde que as conversões bem-sucedidas trouxessem retornos massivos.

### Insight 2: Velocidade de Aprendizado (*Cold Start*)
O Thompson Sampling mitigou o problema de início do experimento de forma exemplar. Enquanto políticas tradicionais exigem longos períodos de testes A/B puros (gerando alto custo de oportunidade), o Thompson Sampling começou a gerar retornos expressivos já nos primeiros 20 passos, otimizando o orçamento do experimento desde o primeiro dia.

### Insight 3: Minimização de Desperdício (*Regret*)
A redução de quase 10.000 unidades de arrependimento pelo Thompson Sampling prova que a alocação dinâmica de tráfego poupou o sistema de exibir opções irrelevantes ou de baixo valor para os usuários, melhorando a experiência geral do cliente e a eficiência operacional.

---

## 6. Conclusões e Recomendações

1.  **Substituição Imediata da Baseline:** Recomenda-se a descontinuação da política fixa e a implementação do Thompson Sampling em ambiente de produção para os cenários de recomendação, precificação ou exibição de ofertas estudados.
2.  **Alinhamento de KPIs de Negócio:** Equipes de produto e marketing devem mover seus KPIs principais de "Taxa de Conversão Bruta" para "Valor de Recompensa por Usuário" (ou receita média por usuário - ARPU). Focar exclusivamente em conversão pode mascarar perdas financeiras severas, como demonstrado pela Baseline.
3.  **Expansão do Modelo:** Sugere-se testar variações contextualizadas do Thompson Sampling (Contextual Bandits), onde variáveis do perfil do usuário e do ambiente histórico possam refinar ainda mais a taxa de conversão sem abrir mão do alto valor médio de recompensa já conquistado.
