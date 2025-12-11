import random
import sys
import time
from datetime import datetime

# Debug Helper
def log(msg):
    """Prints a message with a timestamp to stderr."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", file=sys.stderr)

log("1/6 Starting script...")

# Imports can sometimes be slow if caches are being built
try:
    log("2/6 Importing libraries...")
    from datasets import load_dataset
    from transformers import pipeline
    import torch # importing just to check version if needed
    log("    > Libraries imported successfully.")
except ImportError as e:
    log(f"!!! CRITICAL IMPORT ERROR: {e}")
    sys.exit(1)

# --- CONFIGURATION ---
MODEL_ID = "cross-encoder/nli-deberta-v3-large"
CANDIDATE_LABELS = ["difficult"] 
HYPOTHESIS_TEMPLATE = "This task is {}."
DEVICE = -1 # -1 for CPU (Mac), 0 for GPU (Linux/Windows with NVIDIA)

# --- 1. SETUP PIPELINE ---
log(f"3/6 Initializing Pipeline with model: {MODEL_ID}")
log("    > This step downloads the model (approx 1.7GB) if not cached.")
log("    > Please wait...")

try:
    classifier = pipeline(
        "zero-shot-classification", 
        model=MODEL_ID, 
        device=DEVICE 
    )
    log("    > Pipeline loaded successfully.")
except Exception as e:
    log(f"!!! ERROR loading pipeline: {e}")
    log("    (Did you run 'pip install torch'?)")
    sys.exit(1)

# --- 2. DATA LOADING ---
def get_benchmark_prompts(n_total=40, start_index=0):
    if n_total <= 0: return []
    
    n_math = max(1, round(n_total * 0.20)) if n_total >= 5 else 1
    n_general = n_total - n_math
    
    log(f"4/6 Loading {n_total} prompts (General: {n_general}, Math: {n_math})...")
    all_prompts = []
    random.seed(42)

    # Load GSM8K
    try:
        ds_math = load_dataset("gsm8k", "main", split="test")        
        math_pool = [{"text": row['question'], "cat": "math"} for i, row in enumerate(ds_math) if i >= start_index]
        if math_pool: 
            samples = random.sample(math_pool, min(n_math, len(math_pool)))
            all_prompts.extend(samples)
            log(f"    > Loaded {len(samples)} math prompts.")
    except Exception as e: 
        log(f"    > ⚠️ GSM8k Warning: {e}")

    # Load Alpaca
    try:
        url = "https://huggingface.co/datasets/tatsu-lab/alpaca_eval/resolve/main/alpaca_eval.json"
        ds_alpaca = load_dataset("json", data_files={"test": url}, split="test")
        alpaca_pool = [{"text": row['instruction'], "cat": "general"} for i, row in enumerate(ds_alpaca) if i >= start_index]
        if alpaca_pool: 
            samples = random.sample(alpaca_pool, min(n_general, len(alpaca_pool)))
            all_prompts.extend(samples)
            log(f"    > Loaded {len(samples)} general prompts.")
    except Exception as e: 
        log(f"    > ⚠️ Alpaca Warning: {e}")

    random.shuffle(all_prompts)
    return all_prompts

# --- 3. RUN CLASSIFICATION ---
def run_classification():
    prompts = get_benchmark_prompts(n_total=10)
    
    if not prompts:
        log("!!! No prompts loaded. Exiting.")
        return

    log(f"5/6 Starting classification on {len(prompts)} items...")
    
    # Print Header for standard output
    print(f"\n{'CATEGORY':<10} | {'SCORE':<8} | {'BUCKET':<6} | {'PROMPT SNIPPET'}")
    print("-" * 80)

    for i, row in enumerate(prompts):
        text = row['text']
        
        # Progress indicator in stderr so it doesn't mess up the table
        sys.stderr.write(f"\r    > Processing prompt {i+1}/{len(prompts)}... ")
        sys.stderr.flush()
        
        try:
            start = time.time()
            # INFERENCE STEP
            res = classifier(
                text, 
                candidate_labels=CANDIDATE_LABELS, 
                hypothesis_template=HYPOTHESIS_TEMPLATE,
                multi_label=True 
            )
            
            difficulty_score = res['scores'][0]
            bucket = "EASY" if difficulty_score < 0.80 else "HARD"
            end = time.time()
            latency = end - start
            
            # Print result to standard out
            snippet = text.replace('\n', ' ')
            # [:50] + "..."
            print(f"{row['cat']:<10} | {difficulty_score:.2f}     | {bucket:<6} | {snippet} | {latency}")
            
        except Exception as e:
            print(f"\n!!! Error processing row {i}: {e}")

    sys.stderr.write("\n") # New line after the progress counter
    log("6/6 Done.")

if __name__ == "__main__":
    run_classification()