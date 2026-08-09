import mlflow

from llm.gemini_model import ask_gemini
from llm.groq_model import ask_groq
from rag.prompt_builder import build_rag_prompt

@mlflow.trace
def generate_rag_prompt(state):
    return {
        "rag_prompt": build_rag_prompt(state["prompt"])
    }

@mlflow.trace
def generate_gemini_output(state):
    try:
        output = ask_gemini(state["rag_prompt"])
        return {
            "output": output
        }
    except Exception as e:
        print(f"\n{e}\nSomething wrong with Gemini.\n")
        return {
            "output": "Trying Groq..."
        }

@mlflow.trace
def generate_groq_output(state):
    try:
        output = ask_groq(state["rag_prompt"])
        return {
            "output": output
        }
    except Exception as e:
        print(f"\n{e}\nSomething wrong with Groq.\n")
        return {
            "output": "Error occurred while generating Groq output."
            ""
        }

def check_gemini_output(state):
    if "Trying Groq..." in state["output"]:
        return "call_groq"
    else:
        return "finish"
