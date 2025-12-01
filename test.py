import sys
import os
import time
import numpy as np
from dotenv import load_dotenv

# Import your actual logic
from model_config import MODEL_CATALOG, VALID_CATEGORIES
from models import call_model
from data_loader import get_benchmark_prompts
from main import classify_and_assess # Reusing your actual assessment logic

load_dotenv()

# --- CONSTANTS (Must match main.py) ---
REWARD_PER_QUERY = 0.15
TIME_PENALTY_PER_SEC = 0.004

def run_debug():
    print("\n🔍 LOADING 1 REAL PROMPT...")
    prompts = get_benchmark_prompts(n_total=1)
    if not prompts:
        print("❌ No prompts loaded.")
        return

    text = prompts[0]['text']
    print(f"\n📝 PROMPT: {text[:60]}... ({len(text)} chars)")
    
    print(f"\n{'MODEL':<35} | {'CONF':<6} | {'TOKS':<6} | {'COST':<9} | {'EXP.VAL':<9} | {'RESULT'}")
    print("-" * 100)

    for model_id, specs in MODEL_CATALOG.items():
        provider = specs['provider']
        
        # 1. LIVE API CALL (The part I missed before)
        try:
            # This calls the model to get confidence & tokens
            conf, cat, est_tokens = classify_and_assess(model_id, provider, text)
        except Exception as e:
            print(f"{model_id:<35} | ❌ API ERROR: {e}")
            continue

        # 2. CALCULATE COSTS (Exact logic from main.py)
        # Input length approx len(text)
        input_len = len(text)
        
        # Compute Cost
        cost_compute = (input_len/4 * specs["cost_in"]) + (est_tokens * specs["cost_out"])
        
        # Latency Cost
        lat_est = specs["base_latency"] + (est_tokens / specs["tps"])
        cost_time = lat_est * TIME_PENALTY_PER_SEC
        
        total_cost = cost_compute + cost_time

        # 3. CALCULATE VALUE
        # Assume neutral skills (alpha=1, beta=1) -> mean = 0.5
        # prob = mean * conf
        prob = 0.5 * conf 
        
        # The equation causing "No Bid"
        expected_value = (prob * REWARD_PER_QUERY) - total_cost
        
        bid = max(0.0, expected_value)
        
        # 4. PRINT RESULT
        status = f"✅ ${bid:.4f}" if bid > 0 else "❌ NO BID"
        
        # Highlight why:
        if bid == 0:
            if total_cost > (prob * REWARD_PER_QUERY):
                status += " (Cost > Value)"
            elif conf < 0.1:
                status += " (Low Conf)"

        print(f"{model_id:<35} | {conf:<6.2f} | {est_tokens:<6} | ${total_cost:<8.4f} | ${expected_value:<8.4f} | {status}")

if __name__ == "__main__":
    run_debug()