from llm.gemini_model import ask_gemini
from llm.groq_model import ask_groq
from rag.prompt_builder import build_rag_prompt

def generate_rag_prompt(state):
    return {
        "rag_prompt": build_rag_prompt(state["prompt"])
    }

def generate_model_output(state):
    try:
        output = ask_gemini(state["rag_prompt"])
        return {
            "output": output
        }
    except Exception as e:
        print(f"Error generating model output --- Trying Groq...")
        output = ask_groq(state["rag_prompt"])
        return {
            "output": output
        }
