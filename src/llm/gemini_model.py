import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)

def ask_model(prompt, model="gemini-3.5-flash"):
    """
    Função para interagir com o modelo Gemini.
    """
    response = client.models.generate_content(
        model=model,
        contents=[prompt]
    )
    return response
