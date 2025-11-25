import random
from datasets import load_dataset

def get_benchmark_prompts(n_total=20):
    print(f"--- LOADING {n_total} PROMPTS ---")
    
    prompts = []
    half = n_total // 2

    # 1. LOAD ALPACA EVAL (General Instruction)
    # ROBUST FIX: We load the JSON file directly from the URL. 
    # This bypasses the broken "dataset script" in the repo.
    print(" > Fetching General Instructions (AlpacaEval)...")
    url = "https://huggingface.co/datasets/tatsu-lab/alpaca_eval/resolve/main/alpaca_eval.json"
    
    try:
        # Load directly as JSON. No scripts. No security errors.
        ds_alpaca = load_dataset("json", data_files={"test": url}, split="test")
        
        # Shuffle specifically for Alpaca to get random variety (it's ordered by ID)
        ds_alpaca = ds_alpaca.shuffle(seed=42)
        
        count = 0
        for row in ds_alpaca:
            if count >= half: break
            prompts.append(row['instruction'])
            count += 1
            
    except Exception as e:
        print(f"CRITICAL ERROR loading Alpaca: {e}")
        # Fallback: Just use simple hardcoded prompts if the internet fails
        prompts.append("Write a poem about rust.")
        prompts.append("Explain how a CPU works.")

    # 2. LOAD GSM8k (Math/Logic)
    # GSM8k is hosted by a trusted org and natively supports Parquet, so it works standard.
    print(" > Fetching Logic/Math Problems (GSM8k)...")
    try:
        ds_math = load_dataset("gsm8k", "main", split="test")
        
        # Shuffle GSM8k to avoid getting only "Question 1, Question 2..."
        ds_math = ds_math.shuffle(seed=42)
        
        count = 0
        for row in ds_math:
            if count >= half: break
            prompts.append(row['question'])
            count += 1
            
    except Exception as e:
        print(f"CRITICAL ERROR loading GSM8k: {e}")
        prompts.append("What is 25 * 4?")
        prompts.append("Solve 3x + 5 = 20")

    # 3. Final Shuffle to mix Math and Chat together
    random.shuffle(prompts)
    
    print(f" > Loaded {len(prompts)} prompts.")
    return prompts

if __name__ == "__main__":
    # Self-test
    p = get_benchmark_prompts(10)
    for i, x in enumerate(p):
        print(f"\n{i+1}. {x[:100]}...")