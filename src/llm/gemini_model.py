import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

def ask_gemini(prompt, model="gemini-3.5-flash"):
    """
    Função para interagir com o modelo Gemini.
    """
    if gemini_client is None:
        raise ValueError("GEMINI_API_KEY não configurada.")

    response = gemini_client.models.generate_content(
        model=model,
        contents=[prompt]
    )
    return response.text
