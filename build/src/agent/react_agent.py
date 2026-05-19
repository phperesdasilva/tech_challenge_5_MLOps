# src/agent/react_agent.py
"""Agente ReAct com tools customizadas para o domínio do Datathon.
Referência: Yao et al. (2023) — ReAct: Synergizing Reasoning and Actin
g
in Language Models. https://arxiv.org/abs/2210.03629
"""
import logging
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.tools import Tool

logger = logging.getLogger(__name__)

REACT_PROMPT = PromptTemplate.from_template("""Você é um assistente es
pecializado.
Use as ferramentas disponíveis para responder perguntas.
Ferramentas disponíveis:
{tools}
Use o formato:
Thought: pensar sobre o que fazer
Action: nome_da_ferramenta
Action Input: input para a ferramenta
Observation: resultado da ferramenta
... (repita Thought/Action/Observation quantas vezes necessário)
Thought: Agora sei a resposta final
Final Answer: resposta para o usuário
Pergunta: {input}
{agent_scratchpad}""")

def create_datathon_agent(
    tools: list[Tool],
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    ) -> AgentExecutor:
    
    """Cria agente ReAct para o Datathon.
    Args:
    tools: Lista de ferramentas (≥ 3 obrigatório).
    model_name: Modelo LLM a utilizar.
    temperature: Temperatura de geração.
    Returns:
    AgentExecutor configurado.
    """

    if len(tools) < 3:
        logger.warning("Datathon exige ≥ 3 tools. Fornecidas: %d", len
        (tools))
        llm = ChatOpenAI(model=model_name, temperature=temperature)
        agent = create_react_agent(llm=llm, tools=tools, prompt=REACT_PROMPT)

    return AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=10,
    handle_parsing_errors=True,
    )
