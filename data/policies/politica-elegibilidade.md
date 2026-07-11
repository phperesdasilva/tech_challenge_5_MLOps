# Política de Elegibilidade de Ofertas (SINTÉTICA)

> Documento sintético criado para o Datathon. Não representa regras comerciais reais.

## Regras transversais
- Nenhuma oferta pode usar **renda, gênero, raça, religião ou patrimônio** como critério de decisão.
- Decisões de **crédito e refinanciamento** exigem **humano no loop** antes da efetivação.
- O cliente deve estar **elegível** (regras abaixo) para que o braço entre no conjunto de ações do bandit.

## Ofertas (braços)
- **arm 0 — Conta Digital Padrão** (`baseline`): idade ≥ 18. Sem exigência de financiamento. Público geral.
- **arm 1 — Cartão de Crédito Premium** (`credit`): idade ≥ 21. Requer análise de crédito interna sintética; não ofertar a clientes marcados como inadimplentes.
- **arm 2 — Refinanciamento Imobiliário** (`loan`): idade ≥ 25 **e** possuir financiamento habitacional ativo (`housing = yes`). Não elegível sem financiamento.
- **arm 3 — Depósito a Prazo (CDB)** (`investment`): idade ≥ 18. Público geral. Indicado para perfil conservador/moderado.

## Cooldown
- Oferta recusada não deve ser reapresentada por **7 dias** (janela de cooldown sintética).
