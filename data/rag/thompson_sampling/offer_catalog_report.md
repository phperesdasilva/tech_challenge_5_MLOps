# Relatório de Análise: Catálogo de Ofertas e Performance (MAB - Thompson Sampling)

## 1. Introdução

Este relatório apresenta uma análise detalhada do catálogo de ofertas (versão `1.0.0`) utilizado no motor de recomendação baseado no algoritmo de **Thompson Sampling**. O objetivo deste documento é descrever as características de cada oferta, suas regras de elegibilidade, o balanço entre probabilidade de conversão (*prior*) e valor de recompensa, bem como servir de referência para a análise de desempenho consolidada no arquivo `data/experiments/thompson_sampling/metrics_summary.csv`.

O algoritmo de Thompson Sampling utiliza os dados históricos de conversão para equilibrar a exploração (*exploration*) e a explotação (*exploitation*) das ofertas, otimizando o retorno financeiro esperado por usuário elegível.

---

## 2. Visão Geral do Catálogo de Ofertas

O catálogo é composto por 4 ofertas distintas, cobrindo diferentes perfis de produtos financeiros (serviços básicos, crédito, empréstimos e investimentos).

| ID do Braço | Nome da Oferta | Tipo de Oferta | Taxa de Conversão Prévia (*Prior*) | Valor da Recompensa ($) | Valor Esperado Inicial ($) | Idade Mínima | Requer Empréstimo Habitacional |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **0** | Conta Digital Padrão | `baseline` | 15.0% | 10.00 | 1.50 | 18 anos | Não |
| **1** | Cartão de Crédito Premium | `credit` | 5.0% | 150.00 | 7.50 | 21 anos | Não |
| **2** | Refinanciamento Imobiliário | `loan` | 8.0% | 300.00 | 24.00 | 25 anos | Sim |
| **3** | Depósito a Prazo (CDB) | `investimento` | 12.0% | 80.00 | 9.60 | 18 anos | Não |

*Nota: O Valor Esperado Inicial é calculado como $Prior \times Recompensa$, representando a expectativa matemática de retorno antes da otimização pelo algoritmo.*

---

## 3. Análise Individual das Ofertas

### Braço 0: Conta Digital Padrão
* **Tipo:** `baseline`
* **Descrição:** Abertura de conta corrente digital sem taxas.
* **Perfil de Público:** Clientes jovens ou novos entrantes no ecossistema (Idade $\ge$ 18).
* **Análise Estratégica:** Possui a maior probabilidade de conversão inicial (15.0%), porém o menor valor de recompensa (10.00). É uma oferta porta de entrada, excelente para atrair novos clientes ao portfólio (aquisição), mas com baixo retorno financeiro imediato por conversão.

### Braço 1: Cartão de Crédito Premium
* **Tipo:** `credit`
* **Descrição:** Cartão de crédito com programa de recompensas exclusivo.
* **Perfil de Público:** Clientes com perfil de consumo ativo (Idade $\ge$ 21).
* **Análise Estratégica:** Apresenta a menor taxa de conversão a priori (5.0%), refletindo a maior fricção na aprovação e aceitação de crédito. No entanto, o valor de recompensa é alto (150.00). O algoritmo precisará balancear o baixo volume de aceitação com a alta rentabilidade unitária.

### Braço 2: Refinanciamento Imobiliário
* **Tipo:** `loan`
* **Descrição:** Taxas reduzidas para clientes que já possuem financiamento habitacional ativo.
* **Perfil de Público:** Clientes de perfil maduro e consolidado (Idade $\ge$ 25) que possuem o flag `requer_emprestimo_habitacional` como verdadeiro.
* **Análise Estratégica:** É a oferta mais restritiva devido às regras de elegibilidade cumulativas. Possui o maior valor de recompensa do catálogo (300.00) e uma conversão a priori moderada (8.0%). Embora o valor esperado teórico inicial seja o maior (24.00), a elegibilidade restrita limita o volume absoluto de exibições que esta oferta pode receber.

### Braço 3: Depósito a Prazo (CDB)
* **Tipo:** `investimento`
* **Descrição:** Investimento de renda fixa atrelado ao comportamento e perfil da base original.
* **Perfil de Público:** Clientes com capacidade de poupança (Idade $\ge$ 18).
* **Análise Estratégica:** É uma oferta altamente equilibrada. Apresenta uma taxa de conversão prévia expressiva (12.0%) aliada a uma recompensa intermediária atraente (80.00), resultando em um valor esperado inicial de 9.60. Esta oferta tende a apresentar forte tração e estabilidade durante as rodadas do Thompson Sampling.

---

## 4. Regras de Elegibilidade e Direcionamento de Público

As regras de elegibilidade atuam como um filtro rígido antes que o algoritmo de Thompson Sampling decida qual oferta exibir.

1. **Filtro de Idade:**
   * **Menores de 21 anos** são elegíveis apenas para as ofertas `0` e `3`.
   * **Entre 21 e 24 anos** tornam-se elegíveis também para a oferta `1`.
   * **A partir de 25 anos**, o cliente ganha acesso a todo o catálogo (sujeito à regra de crédito imobiliário).

2. **Filtro de Financiamento Habitacional:**
   * A oferta `2` é estritamente reservada para clientes que possuem financiamento imobiliário prévio. Este direcionamento evita o desperdício de impressões em clientes fora do perfil demográfico e financeiro correto.

---

## 5. Referência de Métricas do Experimento (`metrics_summary.csv`)

Ao analisar a execução do Thompson Sampling através do sumário de métricas consolidado em `data/experiments/thompson_sampling/metrics_summary.csv`, o analista deve atentar para as seguintes dinâmicas de aprendizado do modelo:

1. **Taxa de Conversão Observada vs. Prior:** Verificar se a taxa de conversão empírica convergiu para os valores de *prior* sintéticos (0.15, 0.05, 0.08, 0.12) ou se houve desvios significativos devido à distribuição demográfica da base.
2. **Distribuição de Alocações (Regret Minimization):** Ofertas com maior retorno esperado real (como a combinação de conversão e recompensa do CDB e do Cartão Premium) devem, progressivamente, receber mais alocações à medida que o algoritmo reduz a incerteza de suas distribuições Beta.
3. **Volume de Elegibilidade:** Avaliar se a oferta `2` (Refinanciamento) teve suas exibições severamente limitadas pela restrição de elegibilidade, mesmo apresentando a maior recompensa unitária.
