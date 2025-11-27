import random
from datasets import load_dataset

def get_benchmark_prompts(n_total=20):
    print(f"--- LOADING {n_total} REAL PROMPTS ---")
    
    labeled_prompts = []
    half = n_total // 2

    # 1. LOAD GSM8k (Math)
    print(" > Fetching Math Problems (GSM8k)...")
    try:
        ds_math = load_dataset("gsm8k", "main", split="test")
        ds_math_list = list(ds_math)
        random.shuffle(ds_math_list)
        
        count = 0
        for row in ds_math_list:
            if count >= half: break
            # STRICT DICT FORMAT
            labeled_prompts.append({"text": row['question'], "cat": "math"})
            count += 1
    except Exception as e:
        print(f"Error loading GSM8k: {e}")

    # 2. LOAD ALPACA (General) - Direct JSON to avoid security errors
    print(" > Fetching General Instructions (AlpacaEval)...")
    url = "https://huggingface.co/datasets/tatsu-lab/alpaca_eval/resolve/main/alpaca_eval.json"
    
    try:
        ds_alpaca = load_dataset("json", data_files={"test": url}, split="test")
        ds_alpaca_list = list(ds_alpaca)
        random.shuffle(ds_alpaca_list)
        
        count = 0
        for row in ds_alpaca_list:
            if count >= half: break
            # STRICT DICT FORMAT
            labeled_prompts.append({"text": row['instruction'], "cat": "general"})
            count += 1
    except Exception as e:
        print(f"Error loading Alpaca: {e}")

    # 3. Final Shuffle
    random.shuffle(labeled_prompts)
    
    print(f" > Loaded {len(labeled_prompts)} prompts.")
    return labeled_prompts