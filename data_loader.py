import random
import sys
import time
from datasets import load_dataset

def get_benchmark_prompts(n_total=10, start_index=0):
    """
    Loads n_total prompts with a fallback mechanism if Hugging Face fails.
    """
    if n_total <= 0:
        return []

    # 1. CALCULATE SPLIT
    n_math = max(1, round(n_total * 0.20)) if n_total >= 5 else 1 # Ensure at least 1 math for small tests
    n_general = n_total - n_math
    
    print(f"--- LOADING {n_total} PROMPTS (General: {n_general}, Math: {n_math}) ---", file=sys.stderr)
    
    all_prompts = []
    random.seed(42)

    # 2. LOAD GSM8K (Math)
    try:
        # We try/except the actual download call
        ds_math = load_dataset("gsm8k", "main", split="test")        
        math_pool = [
            {"text": row['question'], "cat": "math"} 
            for i, row in enumerate(ds_math) if i >= start_index
        ]
        
        n_math_actual = min(n_math, len(math_pool))
        if n_math_actual > 0:
            math_samples = random.sample(math_pool, n_math_actual)
            all_prompts.extend(math_samples)
            
    except Exception as e:
        print(f"⚠️ [WARNING] Failed to load GSM8k: {e}", file=sys.stderr)

    # 3. LOAD ALPACA (General)
    url = "https://huggingface.co/datasets/tatsu-lab/alpaca_eval/resolve/main/alpaca_eval.json"
    try:
        ds_alpaca = load_dataset("json", data_files={"test": url}, split="test")
        
        alpaca_pool = [
            {"text": row['instruction'], "cat": "general"} 
            for i, row in enumerate(ds_alpaca) if i >= start_index
        ]
        
        n_general_actual = min(n_general, len(alpaca_pool))
        if n_general_actual > 0:
            alpaca_samples = random.sample(alpaca_pool, n_general_actual)
            all_prompts.extend(alpaca_samples)

    except Exception as e:
        print(f"⚠️ [WARNING] Failed to load Alpaca: {e}", file=sys.stderr)

    # 5. FINAL SHUFFLE
    random.shuffle(all_prompts)
    
    print(f" > Successfully prepared {len(all_prompts)} prompts.", file=sys.stderr)
    
    return all_prompts