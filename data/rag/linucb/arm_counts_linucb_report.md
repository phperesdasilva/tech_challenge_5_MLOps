# Relatório de Análise de Seleção de Braços: Política LinUCB

## 1. Introdução
Este relatório apresenta a análise do comportamento de seleção de braços (ofertas) pela política **LinUCB (Linear Upper Confidence Bound)** durante o período do experimento. O LinUCB é um algoritmo de *Contextual Multi-Armed Bandit* que seleciona a melhor ação (braço) com base em atributos contextuais do usuário e do ambiente, equilibrando a exploração de novas oportunidades (exploração) com a escolha de ofertas com histórico comprovado de sucesso (explorando).

O objetivo desta análise é compreender a distribuição das decisões do modelo, identificar a oferta mais recomendada e levantar hipóteses de negócio que justifiquem essa distribuição.

---

## 2. Análise Quantitativa das Escolhas
Abaixo estão consolidados os dados de volumetria e representatividade de cada braço (oferta) no experimento conduzido pela política LinUCB:

| Posição | Braço (ID) | Contagem (Escolhas) | Participação (%) |
| :---: | :---: | :---: | :---: |
| 1º | **Arm 3** | 2.136 | 47,25% |
| 2º | **Arm 2** | 1.343 | 29,71% |
| 3º | **Arm 1** | 658 | 14,55% |
| 4º | **Arm 0** | 384 | 8,49% |
| **Total** | - | **4.521** | **100,00%** |

### Distribuição Visual (Proporção)
```
Arm 3 [█████████████████████████████████████████████] 47.25%
Arm 2 [██████████████████████████████] 29.71%
Arm 1 [██████████████▊] 14.55%
Arm 0 [████████] 8.49%
```

---

## 3. Principais Descobertas e Comportamento do Algoritmo

### Dominância do Arm 3 e Arm 2
O **Arm 3** foi o campeão absoluto de recomendações, sendo acionado em **47,25%** dos casos (2.136 vezes). Somado ao **Arm 2** (29,71%), ambos concentram **76,96%** de todas as exibições do experimento.

Esse fenômeno indica que o algoritmo LinUCB rapidamente identificou um forte sinal positivo de recompensa (como cliques ou conversões) associado a estes dois braços quando cruzados com as características dos usuários. Por ser um algoritmo linear baseado em contexto, a alta frequência desses braços sugere que eles possuem ampla aderência para diversos perfis demográficos/comportamentais ou que a maior parte do tráfego do experimento pertencia a um segmento que respondeu muito bem a estas duas opções.

### Menor Relevância do Arm 0 e Arm 1
O **Arm 0** obteve apenas **8,49%** das escolhas (384 vezes). Sob a ótica do LinUCB, isso significa que:
1. O braço apresentou um retorno esperado (payoff) consistentemente inferior aos demais quando testado nos diferentes contextos.
2. A incerteza (limite superior de confiança) associada ao Arm 0 diminuiu conforme ele foi explorado, e o modelo concluiu que o valor esperado dele não justificava novas alocações frente aos resultados sólidos do Arm 3.

---

## 4. Hipóteses de Negócio
Do ponto de vista de negócios e de posicionamento de produto, a preferência massiva pelo **Arm 3** e **Arm 2** pode ser explicada por algumas hipóteses:

1. **Apelo Universal da Oferta (Arm 3):** O Arm 3 pode representar um produto de entrada, um desconto agressivo de grande apelo geral ou um serviço essencial de alta conversão natural.
2. **Alinhamento de Persona Dominante:** Se a maior parte do público que entrou no experimento compartilha de características homogêneas (ex: usuários recorrentes, mesma faixa etária ou mesma faixa de renda), e o Arm 3 é altamente otimizado para essa persona, o LinUCB aprenderá a explorar essa correlação quase que de forma contínua.
3. **Sazonalidade ou Momentum:** O Arm 3 pode estar alinhado com uma necessidade imediata do período do teste (ex: uma campanha temática, oferta de frete grátis, etc.), gerando um CTR (Click-Through Rate) inicial muito alto que retroalimentou positivamente a política.

---

## 5. Resumo das Métricas e Impacto no Negócio
Com base no histórico e no resumo de métricas consolidadas (`metrics_summary.csv`), políticas contextualizadas como o LinUCB tendem a superar abordagens tradicionais (como políticas estáticas ou $\epsilon$-greedy puro) porque não desperdiçam tráfego com ofertas irrelevantes para determinados contextos.

* **Minimização de Regret (Arrependimento):** A rápida convergência de tráfego para os braços 3 e 2 mostra que a política conseguiu otimizar a receita/engajamento acumulado rapidamente, reduzindo o "custo de oportunidade" de mostrar ofertas de menor desempenho (Arm 0 e 1).
* **Taxa de Conversão Esperada:** A distribuição assimétrica é um forte indicativo de sucesso na personalização. Em cenários reais, essa distribuição controlada pelo contexto do LinUCB costuma elevar a taxa de conversão geral (CVR) em comparação com uma divisão uniforme (A/B Test tradicional de 25% para cada braço).

---

## 6. Recomendações para Próximos Passos
1. **Análise de Feature Importance:** Investigar quais variáveis contextuais (ex: idade, histórico de compras, dispositivo) foram determinantes para o LinUCB priorizar o Arm 3.
2. **Renovação de Ofertas Subutilizadas:** O Arm 0 e o Arm 1 apresentam baixo desempenho. Recomenda-se revisar a proposta de valor, imagem ou condições dessas ofertas, ou substituí-las por novos conceitos em um próximo ciclo de testes.
3. **Exploração Dinâmica:** Avaliar se a taxa de exploração ($\alpha$) do LinUCB está ideal ou se o modelo convergiu rápido demais para o Arm 3, potencialmente ignorando subgrupos menores onde o Arm 0 ou 1 poderiam performar melhor.
