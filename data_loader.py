import random
from datasets import load_dataset

def get_benchmark_prompts(n_total=10, start_index=0):
    print(f"--- LOADING {n_total} PROMPTS (Offset: {start_index}) ---")
    
    all_prompts = []

    # 1. LOAD GSM8k (Math)
    try:
        ds_math = load_dataset("gsm8k", "main", split="test")
        for row in ds_math:
            all_prompts.append({"text": row['question'], "cat": "math"})
    except Exception as e:
        print(f"Error loading GSM8k: {e}")

    # 2. LOAD ALPACA (General) - Direct JSON to avoid security errors
    url = "https://huggingface.co/datasets/tatsu-lab/alpaca_eval/resolve/main/alpaca_eval.json"
    try:
        ds_alpaca = load_dataset("json", data_files={"test": url}, split="test")
        for row in ds_alpaca:
            all_prompts.append({"text": row['instruction'], "cat": "general"})
    except Exception as e:
        print(f"Error loading Alpaca: {e}")

    # 3. DETERMINISTIC SHUFFLE (Seed 42)
    # This guarantees Training (0-200) and Testing (500-520) never share questions
    random.seed(42) 
    random.shuffle(all_prompts)
    
    # 4. SLICE
    end_index = start_index + n_total
    if end_index > len(all_prompts):
        print(f"Warning: Looping data (Request {end_index} > Avail {len(all_prompts)})")
        sliced = all_prompts[start_index:]
    else:
        sliced = all_prompts[start_index:end_index]
    
    print(f" > Loaded {len(sliced)} prompts.")
    return sliced