from langgraph.graph import StateGraph, END

from graph.state import AgentState
from graph.nodes import check_gemini_output, generate_rag_prompt, generate_gemini_output, generate_groq_output
from graph.router import llm_router

def build_graph():

    graph = StateGraph(AgentState)

    graph.add_node("generate_rag_prompt", generate_rag_prompt)
    graph.add_node("generate_gemini_output", generate_gemini_output)
    graph.add_node("check_gemini_output", check_gemini_output)
    graph.add_node("generate_groq_output", generate_groq_output)

    graph.set_entry_point("generate_rag_prompt")
    graph.add_edge("generate_rag_prompt", "generate_gemini_output")

    graph.add_conditional_edges(
        "generate_gemini_output",
        llm_router,
        {
            "call_groq": "generate_groq_output",
            "finish": END
        }
    )

    graph.add_edge("generate_groq_output", END)

    return graph.compile()
