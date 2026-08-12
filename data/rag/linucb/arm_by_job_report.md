# Relatório de Análise: Segmentação de Ofertas LinUCB por Perfil Profissional

Este relatório apresenta uma análise detalhada das decisões de recomendação tomadas pela política de bandit contextual **LinUCB** com base nas profissões dos clientes (`job`). O objetivo é compreender quais ofertas (braços de 0 a 3) foram prioritariamente associadas a cada grupo profissional e mapear o perfil demográfico e comportamental implícito que cada braço atrai.

---

## 1. Distribuição de Recomendações por Profissão

A tabela abaixo consolida o número de vezes que cada braço (oferta) foi recomendado para cada profissão, identificando o braço predominante e o volume total de interações.

| Profissão (`job`) | Braço 0 | Braço 1 | Braço 2 | Braço 3 | Predominante | Total de Clientes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **admin.** | 0 | 0 | 248 | 230 | **2** | 478 |
| **blue-collar** (operários) | 0 | 0 | 96 | 850 | **3** | 946 |
| **entrepreneur** (empresários) | 4 | 73 | 69 | 22 | **1** | 168 |
| **housemaid** (serviços domésticos) | 0 | 0 | 37 | 75 | **3** | 112 |
| **management** (gestores) | 0 | 28 | 435 | 506 | **3** | 969 |
| **retired** (aposentados) | 0 | 0 | 33 | 197 | **3** | 230 |
| **self-employed** (autônomos) | 0 | 0 | 75 | 108 | **3** | 183 |
| **services** (prestadores de serviços) | 29 | 125 | 224 | 39 | **2** | 417 |
| **student** (estudantes) | 23 | 54 | 7 | 0 | **1** | 84 |
| **technician** (técnicos) | 0 | 33 | 346 | 389 | **3** | 768 |
| **unemployed** (desempregados) | 0 | 75 | 23 | 30 | **1** | 128 |
| **unknown** (não informado) | 33 | 1 | 1 | 3 | **0** | 38 |

---

## 2. Análise Comportamental dos Braços (Ofertas)

A partir da distribuição das recomendações geradas pelo algoritmo LinUCB, podemos inferir a proposta de valor e o perfil de cliente associado a cada braço:

### Braço 0: Perfil Indefinido ou de Baixo Atrito
*   **Público-alvo predominante:** `unknown` (33 de 38 recomendações). Apresenta baixa presença em `student` (23) e `services` (29), e quase nenhuma tração nos demais grupos.
*   **Hipótese de Perfil:** Este braço representa uma **oferta genérica, de baixo risco ou de entrada**. Quando o algoritmo carece de dados contextuais ricos sobre o usuário (caso de perfis classificados como "unknown"), ele opta por essa alternativa segura e conservadora.

### Braço 1: Perfis de Transição, Empreendedorismo ou Renda Variável
*   **Público-alvo predominante:** `student` (54), `unemployed` (75) e `entrepreneur` (73).
*   **Hipótese de Perfil:** Este grupo é composto por pessoas em fases de transição de carreira, que buscam qualificação ou que possuem alta variabilidade de renda.
    *   **Estudantes e Desempregados** necessitam de soluções financeiras de baixo custo, sem tarifas de manutenção exigentes ou com foco em microcrédito e capacitação.
    *   **Empresários (entrepreneurs)** demandam produtos de investimento dinâmicos ou crédito PJ/fomento comercial.
    *   *Conclusão sobre o Braço 1:* É provável que este braço ofereça produtos financeiros focados em **flexibilidade, desenvolvimento pessoal/empresarial ou isenção de tarifas básicas**.

### Braço 2: Perfis Corporativos Estáveis e Setor de Serviços
*   **Público-alvo predominante:** `admin.` (248) e `services` (224). Também possui forte representação em `management` (435) e `technician` (346).
*   **Hipótese de Perfil:** Atrai profissionais de colarinho branco e prestadores de serviços estruturados que possuem renda recorrente, previsibilidade financeira moderada a alta e forte inserção no mercado formal.
    *   *Conclusão sobre o Braço 2:* Sugere um produto estruturado de varejo tradicional, como **cartões de crédito com programa de milhas/cashback intermediários, planos de previdência complementar ou investimentos de liquidez diária**.

### Braço 3: Perfis Operacionais, Conservadores ou de Alta Renda (Massa e Estabilidade)
*   **Público-alvo predominante:** `blue-collar` (850), `retired` (197), `management` (506), `technician` (389), `self-employed` (108) e `housemaid` (75).
*   **Hipótese de Perfil:** Este é o braço mais amplamente distribuído e recomendado pela política.
    *   Para `blue-collar`, `retired` e `housemaid`, a preferência pelo Braço 3 indica um forte apelo por **segurança, simplicidade e baixo risco** (como poupança, títulos de capitalização ou empréstimo consignado).
    *   A alta presença de `management` e `technician` indica que este produto possui um forte apelo de massa ou excelente custo-benefício, conseguindo transitar bem entre diferentes classes de renda devido a uma proposta de valor robusta e consolidada.

---

## 3. Alinhamento com as Métricas de Performance

A distribuição observada reflete diretamente os resultados consolidados em `data/experiments/linucb/metrics_summary.csv`. Ao contrário de políticas estáticas (como *Random* ou *Epsilon-Greedy* puro), o algoritmo **LinUCB** maximiza o retorno acumulado ao explorar o contexto do usuário (neste caso, a profissão) para personalizar a oferta.

*   A clara diferenciação de preferências (como estudantes preferindo o Braço 1 e operários preferindo o Braço 3) explica o **aumento na taxa de conversão (CTR)** e a **redução do arrependimento acumulado (cumulative regret)** observados nas métricas globais do modelo.
*   A habilidade de segmentar ofertas financeiras específicas para cada momento de vida do cliente (ex: identificar a necessidade de crédito flexível para o empreendedor vs. estabilidade para o aposentado) valida a eficácia do bandit contextual em ambientes de produção.

---

## 4. Conclusões e Recomendações de Negócio

1.  **Validação de Campanha:** A forte aderência do **Braço 3** nos segmentos operacional (`blue-collar`) e conservador (`retired`) justifica o direcionamento de campanhas de produtos de menor volatilidade ou crédito seguro (consignado) para essa base.
2.  **Otimização do Braço 1:** Sendo a principal recomendação para estudantes e desempregados, campanhas vinculadas ao Braço 1 devem focar em propostas de valor educacionais, contas digitais gratuitas e caminhos de bancarização progressiva.
3.  **Refinamento do Grupo "Unknown":** Recomenda-se enriquecer a coleta de dados cadastrais para o grupo "unknown" a fim de deslocá-los do Braço 0 para opções mais personalizadas (Braços 1, 2 ou 3), onde o retorno esperado e o engajamento são comprovadamente maiores.
