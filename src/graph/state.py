from typing import TypedDict

class AgentState(TypedDict):
    prompt: str
    output: str
    function_call: str
    rag_prompt: str
