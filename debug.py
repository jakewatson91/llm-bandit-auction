import os
import time
import sys
from dotenv import load_dotenv

# Client Imports
from groq import Groq
from openai import OpenAI
from google import genai

# --- 1. SETUP ---
load_dotenv()

print("--- 1. INITIALIZING CLIENTS ---")
try:
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    print("[✓] Clients Initialized")
except Exception as e:
    print(f"[X] Init Error: {e}")
    sys.exit(1)

# --- 2. DEFINE TEST DATA (The new structure) ---
# This mimics what will be in model_config.py
TEST_CONFIG = [
    # GROQ: Simple ID
    {"provider": "groq", "model": "llama-3.3-70b-versatile"},
    
    # GROQ: Complex ID (The one that broke the split function)
    {"provider": "groq", "model": "meta-llama/llama-4-maverick-17b-128e-instruct"}, 
    
    # OPENAI
    {"provider": "openai", "model": "gpt-4o-mini"},
    
    # GEMINI
    {"provider": "gemini", "model": "gemini-2.0-flash"}
]

# --- 3. THE NEW ROUTER LOGIC ---
def call_model_explicit(provider, model_id, prompt):
    """
    Takes explicit provider + model_id. No string parsing.
    """
    if provider == "groq":
        resp = groq_client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            timeout=15.0
        )
        return resp.choices[0].message.content

    elif provider == "openai":
        resp = openai_client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            timeout=15.0
        )
        return resp.choices[0].message.content

    elif provider == "gemini":
        resp = gemini_client.models.generate_content(
            model=model_id,
            contents=prompt
        )
        return resp.text
    
    else:
        raise ValueError(f"Unknown Provider: {provider}")

# --- 4. RUN TEST ---
print("\n--- 2. TESTING EXPLICIT ROUTING ---")
print(f"{'PROVIDER':<10} | {'MODEL ID':<50} | {'STATUS'}")
print("-" * 80)

failures = []

for entry in TEST_CONFIG:
    prov = entry["provider"]
    mid = entry["model"]
    
    try:
        start = time.time()
        call_model_explicit(prov, mid, "Say '1'")
        lat = time.time() - start
        print(f"{prov:<10} | {mid:<50} | \033[92mPASS\033[0m ({lat:.2f}s)")
    except Exception as e:
        err = str(e).split('\n')[0][:30]
        print(f"{prov:<10} | {mid:<50} | \033[91mFAIL\033[0m ({err}...)")
        failures.append(mid)

print("-" * 80)
if failures:
    print(f"[!] {len(failures)} Tests Failed.")
else:
    print("[✓] Logic Verified. You can now apply this structure to main files.")