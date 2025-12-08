import sys
import os

# 1. IMMEDIATE PRINT TO VERIFY PYTHON IS RUNNING
print("✅ [1/10] Script started. Python interpreter is working.", flush=True)

try:
    import time
    import argparse
    import random
    import json
    print("✅ [2/10] Standard libraries imported.", flush=True)
except Exception as e:
    print(f"❌ [CRITICAL] Standard library import failed: {e}", flush=True)
    sys.exit(1)

try:
    from dotenv import load_dotenv
    print("✅ [3/10] python-dotenv imported.", flush=True)
except ImportError:
    print("❌ [CRITICAL] 'python-dotenv' not installed. Run: pip install python-dotenv", flush=True)
    sys.exit(1)

try:
    from openai import OpenAI
    print("✅ [4/10] openai library imported.", flush=True)
except ImportError:
    print("❌ [CRITICAL] 'openai' library not installed. Run: pip install openai", flush=True)
    sys.exit(1)

# --- CONFIGURATION ---
GEMMA_CONFIG = { 
    "model_id": "tngtech/deepseek-r1t2-chimera:free",
    "provider": "openrouter",
}

def run_paranoid_test():
    print("\n--- ENTERING MAIN FUNCTION ---", flush=True)
    
    # 2. LOAD ENV
    print("🔍 [5/10] Loading .env file...", flush=True)
    load_dotenv()
    
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ [FAILURE] OPENROUTER_API_KEY is MISSING from environment.", flush=True)
        print(f"   Current Directory: {os.getcwd()}", flush=True)
        print(f"   Env keys found: {list(os.environ.keys())}", flush=True)
        return
    else:
        masked = f"{api_key[:5]}...{api_key[-4:]}"
        print(f"✅ [6/10] API Key found: {masked}", flush=True)

    # 3. INIT CLIENT
    print("🔍 [7/10] Initializing OpenAI Client...", flush=True)
    try:
        client = OpenAI(
            api_key=api_key, 
            base_url="https://openrouter.ai/api/v1"
        )
        print("✅ Client object created.", flush=True)
    except Exception as e:
        print(f"❌ [FAILURE] Client init crashed: {e}", flush=True)
        return

    # 4. PREPARE CALL
    print(f"🔍 [8/10] Preparing to call model: {GEMMA_CONFIG['model_id']}...", flush=True)
    
    max_retries = 3
    for attempt in range(max_retries):
        print(f"\n   --- Attempt {attempt + 1} ---", flush=True)
        try:
            print("   >> Sending Network Request...", flush=True)
            start = time.time()
            
            # THE CALL
            response = client.chat.completions.create(
                model=GEMMA_CONFIG['model_id'],
                messages=[{"role": "user", "content": "Say hello."}],
            )
            
            duration = time.time() - start
            print(f"   << Response received in {duration:.2f}s", flush=True)
            
            content = response.choices[0].message.content
            print("\n✅ [10/10] SUCCESS!", flush=True)
            print("-" * 40)
            print(content)
            print("-" * 40)
            return

        except Exception as e:
            print(f"\n❌ [ERROR] Exception Caught:", flush=True)
            print(f"   Type: {type(e).__name__}", flush=True)
            print(f"   Message: {str(e)}", flush=True)
            
            if attempt < max_retries - 1:
                print("   ⚠️ Waiting before retry...", flush=True)
                time.sleep(1)
            else:
                print("❌ [FAILURE] All retries exhausted.", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    print(f"🔍 Script called with args: {sys.argv}", flush=True)
    
    if args.test:
        run_paranoid_test()
    else:
        print("⚠️  WARNING: You didn't pass --test. Running anyway for debugging...", flush=True)
        run_paranoid_test()