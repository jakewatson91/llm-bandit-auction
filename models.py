import os
from dotenv import load_dotenv
from openai import OpenAI
from groq import Groq
from google import genai
from google.genai import types
import time
import random

# Ensure model_config is correctly imported without CATEGORIES/TEMPERATURES if those files were merged
from model_config import FALLBACK_CONFIG, MAX_API_RETRIES, MODEL_CATALOG

load_dotenv()

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
deepseek_client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
openrouter_client = OpenAI(api_key=os.environ.get("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1")

def _get_client(provider):
    if provider == "openai":
        return openai_client
    elif provider == "groq":
        return groq_client
    elif provider == "gemini":
        return gemini_client
    elif provider == "deepseek":
        return deepseek_client
    elif provider == "openrouter":
        return openrouter_client
    else:
        raise ValueError(f"Unknown provider: {provider}")

def call_model(provider, model, prompt):
    """
    Core atomic function to execute a single, synchronous API call to the specified model.
    Removed 'temp' parameter.
    """
    client = _get_client(provider)

    if client == gemini_client:
        # Using google.genai Client structure
        response = client.models.generate_content(
            model=model, 
            contents=prompt,
        )
        return response.text
    else:
        messages = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        return response.choices[0].message.content
    
def call_model_with_retry(provider, model_id, prompt_text, is_judge=False):
    """
    Calls the LLM API, wrapping the call_model function with exponential 
    backoff, jitter, and fallback logic on technical failure (503/network errors).
    Removed 'temperature' parameter.

    Args:
        provider (str): The initial provider.
        model_id (str): The initial model ID.
        prompt_text (str): The input text.
        is_judge (bool): If True, disables the fallback model switch.
        
    Returns:
        tuple: (response (str or None), latency (float))
               Returns None and 0.0 on total failure.
    """
    max_retries = MAX_API_RETRIES
    response = None
    latency = 0.0

    current_provider = provider
    current_model_id = model_id
    is_judge = MODEL_CATALOG[current_model_id].get('is_judge', False)
    
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            
            # Removed temperature argument
            response = call_model(current_provider, current_model_id, prompt_text)
            
            latency = time.time() - start_time
            return response, latency 
            
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  🛑 API Fail: {current_model_id} failed after {max_retries} retries.")
                return None, 0.0
            
            # --- EXPONENTIAL BACKOFF CALCULATION ---
            delay = (2 ** attempt) + random.uniform(0.1, 0.5) 
            
            print(f"  ⚠️  [API Fail] {current_model_id} crashed (Attempt {attempt+1}). Backing off for {delay:.2f}s...")
            time.sleep(delay)

            # Switch to fallback model if the winner fails the initial attempt (only if not a judge call)
            if attempt == 0 and not is_judge:
                # Assumes FALLBACK_CONFIG is available globally
                current_provider = FALLBACK_CONFIG['provider']
                current_model_id = FALLBACK_CONFIG['model']
                
    return None, 0.0