from llm.gemini_model import ask_model
from rag.prompt_builder import build_rag_prompt

def generate_rag_prompt(state):
    return {
        "rag_prompt": build_rag_prompt(state["prompt"])
    }

def generate_model_output(state):
    output = ask_model(state["rag_prompt"])
    return {
        "output": output.text
    }
