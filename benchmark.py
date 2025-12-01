import pandas as pd
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import call_model
from data_loader import get_benchmark_prompts
from model_config import BENCHMARK_MODELS, MODEL_CATALOG
from main import initialize_market, run_simulation, judge_answer

# --- CONFIGURATION ---
TEST_SIZE = 3
TEST_DATA_OFFSET = 500

def run_baselines(prompts):
    logs = []
    print(f"\n--- RUNNING BASELINES ({len(prompts)} Prompts) ---", flush=True)

    for model_id, specs in BENCHMARK_MODELS.items():
        print(f"\n🔵 Testing Baseline Model: {model_id}", flush=True)
        provider = specs['provider']
        
        # Read throttle delay from config (defaults to 0 if not set)
        throttle_sec = specs.get("throttle_delay", 0)

        for i, p in enumerate(prompts):
            text = p["text"]
            
            # --- THROTTLING ---
            if i > 0 and throttle_sec > 0:
                # time between requests for rate limits
                time.sleep(throttle_sec)
            # --------------------------

            print(f"   [{i+1}/{len(prompts)}] Processing...", flush=True)
            start = time.time()
            try:
                answer = call_model(provider, model_id, text)
                latency = time.time() - start
                
                cost = (len(text) / 4 * specs["cost_in"]) + (len(answer) / 4 * specs["cost_out"])
                score = judge_answer(text, answer)
            except Exception as e:
                print(f"\n   ❌ Failed on prompt {i+1}: {e}", flush=True)
                latency = 0; cost = 0; score = 0

            logs.append({
                "System": f"Baseline ({model_id})",
                "Winner": model_id,
                "Latency": latency,
                "Cost": cost,
                "Score": score
            })
        print(f"   ✅ Completed {len(prompts)} prompts for {model_id}.", flush=True)
    return pd.DataFrame(logs)

def run_market(prompts):
    print(f"\n--- RUNNING MARKET SIMULATION ({len(prompts)} Prompts) ---", flush=True)
    market = initialize_market()
    
    # Requirement #3: No suppression
    raw = run_simulation(market, prompts, evaluate=True)

    rows = []
    for _, r in raw.iterrows():
        # Requirement #4: Simplified Extraction
        rows.append({
            "System": "Market", 
            "Winner": r["winner"],
            "Latency": r.get("latency", 0), 
            "Cost": r.get("compute_cost", 0), 
            "Score": r["score"]
        })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    print(f"Loading {TEST_SIZE} prompts (Offset: {TEST_DATA_OFFSET})...", flush=True)
    prompts = get_benchmark_prompts(n_total=TEST_SIZE, start_index=TEST_DATA_OFFSET)

    df_base = run_baselines(prompts)
    df_market = run_market(prompts)

    combined = pd.concat([df_base, df_market])
    combined.to_csv("benchmark_results.csv", index=False)

    print("\n" + "="*80)
    print("🏁 FINAL BENCHMARK REPORT")
    print("="*80)
    
    # Requirement #5: Simple stats
    if not combined.empty:
        print(combined.groupby("System")[["Cost", "Latency", "Score"]].mean())
    
    print("="*80)
    print("Detailed results saved to 'benchmark_results.csv'")