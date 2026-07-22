def llm_router(state):
    if "Trying Groq..." in state["output"]:
        return "call_groq"
    else:
        return "finish"
