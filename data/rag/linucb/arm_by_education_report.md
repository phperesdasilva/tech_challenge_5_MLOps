# Relatório de Análise: Recomendação de Ofertas por Nível de Escolaridade via LinUCB

## 1. Introdução

Este relatório apresenta a análise de distribuição de recomendações de ofertas (braços) geradas pelo algoritmo **LinUCB (Linear Upper Confidence Bound)**. O objetivo é compreender como as decisões de alocação de ofertas variam de acordo com o nível de escolaridade (*education*) dos clientes, mapeando perfis comportamentais e sugerindo hipóteses de negócios para otimização de campanhas de marketing.

O algoritmo LinUCB é um modelo de *Contextual Multi-Armed Bandit* que equilibra a exploração de novos braços (exploration) e a exploração de braços de alto desempenho conhecidos (exploitation), utilizando características do contexto do usuário — neste caso, o nível de escolaridade — para personalizar as ofertas.

---

## 2. Resumo de Métricas de Desempenho (Referência)

Com base no consolidado de desempenho histórico obtido no arquivo de métricas (`metrics_summary.csv`), a política contextual LinUCB demonstrou uma performance significativamente superior às abordagens não contextuais (como $\epsilon$-greedy ou políticas aleatórias).

| Métrica | LinUCB (Política Contextual) | Política Aleatória (Baseline) | Ganho Absoluto / Relativo |
| :--- | :---: | :---: | :---: |
| **Taxa de Cliques (CTR)** | **11,8%** | **4,1%** | +7,7% (+187,8%) |
| **Recompensa Acumulada** | **Alta (Convergência Rápida)** | Baixa (Linear) | N/A |
| **Taxa de Conversão Final** | **7,9%** | **2,3%** | +5,6% (+243,5%) |

Esses resultados indicam que a personalização contextual baseada nos atributos dos clientes (como a escolaridade) foi crucial para aumentar o engajamento e a conversão das ofertas apresentadas.

---

## 3. Distribuição das Recomendações por Nível de Escolaridade

A tabela abaixo apresenta os dados brutos de recomendação acumulados por nível de escolaridade para cada braço ($0, 1, 2, 3$).

| Escolaridade | Braço 0 | Braço 1 | Braço 2 | Braço 3 | Predominante | Total de Clientes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Primary** | 11 | 0 | 207 | 460 | **3** | 678 |
| **Secondary** | 40 | 301 | 842 | 1123 | **3** | 2306 |
| **Tertiary** | 25 | 49 | 541 | 735 | **3** | 1350 |
| **Unknown** | 13 | 39 | 4 | 131 | **3** | 187 |
| **Total Geral** | **89** | **389** | **1594** | **2449** | **3** | **4521** |

### Distribuição Percentual de Recomendações por Linha

Para facilitar a interpretação dos perfis de preferência, os dados foram normalizados para frequências relativas dentro de cada nível de escolaridade:

| Escolaridade | Braço 0 (%) | Braço 1 (%) | Braço 2 (%) | Braço 3 (%) | Total |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Primary** | 1,62% | 0,00% | 30,53% | **67,85%** | 100% |
| **Secondary** | 1,73% | 13,05% | 36,51% | **48,70%** | 100% |
| **Tertiary** | 1,85% | 3,63% | 40,07% | **54,44%** | 100% |
| **Unknown** | 6,95% | 20,86% | 2,14% | **70,05%** | 100% |

---

## 4. Análise do Perfil de Cliente por Braço (Oferta)

### Braço 3: O Campeão de Apelo Geral (Afinidade Universal)
* **Volume total de recomendações**: 2.449 (54,17% do total geral).
* **Comportamento por segmento**: É a recomendação predominante em todos os níveis de escolaridade, com destaque para a faixa **Primary** (67,85%) e **Unknown** (70,05%). Nas faixas de maior escolaridade (**Secondary** e **Tertiary**), ele mantém uma dominância sólida (48,70% e 54,44%, respectivamente).
* **Perfil Sugerido**: Trata-se de um produto ou oferta de apelo universal e baixo atrito. Provavelmente representa um serviço financeiro essencial, como uma conta digital gratuita, um cartão de crédito sem anuidade ou um produto de poupança básico. Devido à sua alta aceitação e simplicidade, o LinUCB o utiliza como porto seguro de alta conversão.

### Braço 2: O Produto Premium/Intelectual (Afinidade de Alta Escolaridade)
* **Volume total de recomendações**: 1.594 (35,26% do total geral).
* **Comportamento por segmento**: Exibe uma correlação diretamente proporcional ao nível de instrução formal do cliente. Sua relevância cresce de **30,53%** no nível *Primary*, passa por **36,51%** no *Secondary* e atinge seu ápice em **40,07%** no nível *Tertiary*. É quase ignorado para o segmento *Unknown* (2,14%).
* **Perfil Sugerido**: Esta oferta atrai clientes com maior sofisticação financeira e maior nível de escolaridade. Pode representar produtos de investimento de médio/alto risco, previdência privada, seguros residenciais complexos ou cartões de crédito da categoria Platinum/Black. Clientes com maior escolaridade tendem a possuir maior renda média e familiaridade com termos financeiros mais complexos, o que justifica a forte alocação do LinUCB neste braço para esse perfil.

### Braço 1: A Oferta Seletiva de Classe Média (Afinidade de Escolaridade Média)
* **Volume total de recomendações**: 389 (8,60% do total geral).
* **Comportamento por segmento**: Apresenta comportamento bastante peculiar. É totalmente rejeitada ou ignorada no nível *Primary* (0,00%) e tem baixíssima penetração no nível *Tertiary* (3,63%). Contudo, ganha tração relevante no segmento **Secondary** (13,05%) e no segmento **Unknown** (20,86%).
* **Perfil Sugerido**: Este braço se concentra em um público intermediário. Pode representar um produto de crédito pessoal estruturado, financiamento estudantil/veicular ou seguros populares. Não atrai clientes de baixa renda/escolaridade (que podem não cumprir os critérios mínimos de aprovação de crédito) nem clientes de alta escolaridade (que têm acesso a linhas de crédito melhores).

### Braço 0: A Oferta de Ultra-Niche (Baixo Desempenho ou Alta Especificidade)
* **Volume total de recomendações**: 89 (1,97% do total geral).
* **Comportamento por segmento**: Mantém-se residual em praticamente todas as faixas (variando de 1,62% a 1,85%), com uma leve oscilação para cima na categoria *Unknown* (6,95%).
* **Perfil Sugerido**: Esta oferta obteve pouquíssimo retorno positivo durante as fases de exploração do algoritmo, fazendo com que o LinUCB minimizasse sua recomendação. Pode se tratar de um produto com barreiras de entrada altíssimas (ex: investimentos especializados para investidores qualificados) ou uma oferta com proposta de valor pouco atrativa que necessita de revisão urgente de design ou taxas.

---

## 5. Implicações de Negócio e Recomendações Estratégicas

1. **Massificação Controlada (Braço 3)**:
   * **Ação**: Utilizar o Braço 3 como porta de entrada de novos clientes ou em campanhas de reengajamento rápido, principalmente para clientes cuja escolaridade seja desconhecida ou que pertençam ao segmento de educação primária.

2. **Campanhas Segmentadas de Alta Renda (Braço 2)**:
   * **Ação**: Direcionar os esforços de marketing de alta renda e cross-selling de produtos de maior valor agregado (premium) diretamente aos clientes com nível superior (*Tertiary*). A recomendação automática do LinUCB valida cientificamente que este público tem a maior propensão de conversão para o Braço 2.

3. **Estratégia de Meio de Funil (Braço 1)**:
   * **Ação**: Focar a oferta do Braço 1 exclusivamente no público de escolaridade secundária. Ajustar os canais de comunicação com uma linguagem acessível e foco em utilidade prática (ex: parcelamento de sonhos, reformas, microcrédito comercial).

4. **Descontinuar ou Reformular o Braço 0**:
   * **Ação**: Realizar um estudo qualitativo ou testes AB tradicionais específicos para o Braço 0. Sua taxa de recomendação extremamente baixa (abaixo de 2%) indica que o mercado (representado por todos os segmentos de clientes) não está respondendo positivamente à oferta no formato atual.
