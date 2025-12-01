import os
from dotenv import load_dotenv
from openai import OpenAI
from groq import Groq
from google import genai

load_dotenv()

# -------------------------
# Clients
# -------------------------

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# -------------------------
# Single Unified Call
# -------------------------

def call_model(provider, model, prompt, system=None):
    """
    provider: "openai" | "groq" | "gemini"
    model: model id string
    prompt: user prompt
    system: optional system prompt
    """

    if provider == "gemini":
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        # Using google.genai Client structure
        response = gemini_client.models.generate_content(
            model=model, 
            contents=full_prompt
        )
        return response.text

    if provider == "openai":
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = openai_client.chat.completions.create(
            model=model,
            messages=messages
        )
        return response.choices[0].message.content

    if provider == "groq":
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = groq_client.chat.completions.create(
            model=model,
            messages=messages
        )
        return response.choices[0].message.content

    raise ValueError(f"Unknown provider: {provider}")