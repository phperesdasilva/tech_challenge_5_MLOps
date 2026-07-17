import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

groq_client = Groq(api_key=GROQ_API_KEY)

def ask_groq(prompt, model="llama-3.3-70b-versatile"):
    """
    Função para interagir com o modelo Groq (ex: LLaMA 3).
    """
    response = groq_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ]
    )

    return response.choices[0].message.content
