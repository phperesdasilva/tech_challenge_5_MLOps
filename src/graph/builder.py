from langgraph.graph import StateGraph, END

from graph.state import AgentState
from graph.nodes import generate_rag_prompt, generate_model_output

def build_graph():

    graph = StateGraph(AgentState)

    graph.add_node("generate_rag_prompt", generate_rag_prompt)
    graph.add_node("generate_model_output", generate_model_output)

    graph.set_entry_point("generate_rag_prompt")
    graph.add_edge("generate_rag_prompt", "generate_model_output")
    graph.add_edge("generate_model_output", END)

    return graph.compile()
