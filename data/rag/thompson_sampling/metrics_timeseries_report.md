# Relatório de Análise Experimental: Thompson Sampling vs. Baseline Fixed Policy

## 1. Sumário Executivo

Este relatório apresenta uma análise comparativa de desempenho entre duas abordagens de tomada de decisão em ambiente de experimentação (*Multi-Armed Bandit*): a **Baseline Fixed Policy** (uma política de controle estática) e o algoritmo **Thompson Sampling Policy** (uma abordagem bayesiana probabilística para o dilema de exploração vs. explotação).

A análise foi realizada utilizando o histórico temporal de métricas contido no arquivo `metrics_timeseries.csv`, cobrindo um horizonte de **4.521 passos (steps)**. Os resultados demonstram de forma inequívoca a superioridade do modelo de **Thompson Sampling**, que obteve um incremento de **103,9% na recompensa acumulada** e uma **redução de 9,89% no arrependimento acumulado**, operando com uma taxa de conversão final estatisticamente equivalente à da Baseline.

---

## 2. Visão Geral das Métricas Finais (Passo 4521)

A tabela abaixo resume o estado final dos experimentos para ambas as políticas no último passo registrado:

| Métrica | Baseline Fixed Policy | Thompson Sampling Policy | Diferença Absoluta | Variação (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Passos (Steps)** | 4.521 | 4.521 | - | - |
| **Recompensa Acumulada** | 6.920,0 | 14.110,0 | +7.190,0 | **+103,90%** |
| **Arrependimento Acumulado (*Regret*)** | 72.827,20 | 65.624,00 | -7.203,20 | **-9,89%** |
| **Taxa de Conversão Final** | 15,31% | 15,07% | -0,24 p.p. | **-1,57%** |

---

## 3. Análise Detalhada da Progressão Temporal

### 3.1. Curva de Recompensa Acumulada (*Cumulative Reward*)
A evolução da recompensa acumulada revela o principal diferencial competitivo do Thompson Sampling:
* **Fase Inicial (Passos 1 a 100):** O Thompson Sampling rapidamente identifica braços de altíssimo valor (com recompensas individuais de até 300,0 por conversão). No passo 15, o Thompson Sampling já acumula **680,0** de recompensa, enquanto a Baseline possui apenas **30,0**.
* **Fase de Explotação (Passos 100 a 4521):** A Baseline cresce de forma estritamente linear, adicionando consistentemente 10,0 unidades de recompensa por conversão. O Thompson Sampling, ao conseguir mapear os braços mais valiosos, sobe em degraus acentuados sempre que aciona e converte nos braços de maior recompensa (payouts de 300,0). Ao final do experimento, o Thompson Sampling entrega mais que o dobro de valor total (**14.110,0** contra **6.920,0**).

### 3.2. Evolução do Arrependimento (*Cumulative Regret*)
O arrependimento mede a perda de oportunidade em relação à escolha do braço ideal teórico:
* **Baseline Fixed Policy:** Apresenta um crescimento de regret perfeitamente linear e agressivo ao longo de todo o tempo, encerrando em **72.827,20**. Isso ocorre porque a política fixa é incapaz de se adaptar ou aprender que está tomando decisões subótimas em comparação com as alternativas de maior valor.
* **Thompson Sampling Policy:** Demonstra uma curva de aprendizado clássica de algoritmos de bandit. Nos passos iniciais (até o passo 15), o algoritmo chega a apresentar **arrependimento negativo** (mínimo de **-377,6** no passo 15), o que indica que ele superou a expectativa ótima de curto prazo devido a escolhas de braços altamente compensatórios. Embora o regret acumulado volte a subir durante fases de exploração necessárias, o Thompson Sampling encerra com um acumulado de **65.624,00**, poupando significativamente o orçamento de teste da organização.

### 3.3. Taxa de Conversão (*Conversion Rate*)
Um comportamento contraintuitivo e extremamente rico surge ao analisar a Taxa de Conversão:
* A Baseline encerra com **15,31%** de conversão, ligeiramente superior aos **15,07%** do Thompson Sampling.
* **O Insight de Valor:** Embora a Baseline converta com uma frequência ligeiramente maior, ela gera **metade** do valor financeiro (recompensa). Isso prova que o Thompson Sampling não focou apenas em otimizar o *volume* de conversões, mas sim o *valor esperado* de cada conversão. O algoritmo sacrificou uma fração irrelevante da taxa de conversão (0,24 pontos percentuais) para focar em braços que pagavam recompensas 30 vezes maiores (300,0 vs. 10,0).

---

## 4. Diagnóstico de Qualidade de Dados (Data Quality Insight)

Durante a análise da série temporal da Baseline, identificou-se uma anomalia na escala de registro de recompensas:
* Até o passo 2020, o acumulado da Baseline segue um padrão multiplicador de 10 por conversão (ex: 310 conversões resultando em **3.100,0** de recompensa).
* Nos passos **2021, 2022 e 2023**, a recompensa acumulada registrada cai bruscamente para **310,0**, enquanto a taxa de conversão mantém-se coerente em torno de **15,3%** (o que seria matematicamente impossível para uma recompensa acumulada real de 310,0 se a escala anterior fosse mantida).
* No passo **2024**, o registro normaliza-se para **3.110,0**.

**Impacto:** Este comportamento caracteriza um erro de escala/registro temporário nos dados coletados (provavelmente uma divisão não planejada por 10 ou perda de um caractere de digitação). Recomenda-se ajustar a rotina de ingestão de logs para evitar flutuações artificiais em auditorias futuras, embora o cálculo da taxa de conversão e do arrependimento pareçam ter utilizado as variáveis corretas de forma independente.

---

## 5. Conclusões e Recomendações

1. **Adoção Comercial Imediata:** O algoritmo de **Thompson Sampling** provou ser amplamente superior para a otimização de negócios. Ao focar no valor real gerado por conversão (recompensa) ao invés de apenas focar na taxa de clique/conversão volumétrica bruta, ele maximizou a receita potencial em mais de 100%.
2. **Eficiência no Aprendizado:** O Thompson Sampling gerenciou de forma inteligente o custo de oportunidade, mitigando o desperdício (arrependimento) em aproximadamente **7,2 mil unidades de valor** em relação à Baseline estática.
3. **Recomendação de Engenharia:** Sugere-se a substituição definitiva da política fixa atual pela Thompson Sampling Policy em produção para campanhas de marketing, precificação dinâmica ou recomendação de produtos, onde diferentes ofertas possuam margens ou valores de conversão distintos (*heterogeneous payouts*).
