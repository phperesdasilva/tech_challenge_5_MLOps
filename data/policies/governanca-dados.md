# Governança de Dados do Assistente (SINTÉTICA)

> Resumo sintético das travas de dados que o assistente deve respeitar.

## Dados proibidos para decisão
Renda, salário, gênero, sexo, raça, cor, etnia, religião, orientação sexual e patrimônio **não podem** ser usados como critério de oferta ou recomendação.

## Princípios
- **Minimização**: usar apenas o contexto necessário para a decisão.
- **Humano no loop**: decisões sensíveis (crédito, refinanciamento) não são efetivadas de forma totalmente automática.
- **Rastreabilidade**: toda decisão e resposta do assistente registra reason codes e a versão de política aplicada.
- **Retenção**: logs de decisão e telemetria seguem ciclo de retenção sintético definido no plano LGPD do projeto.
