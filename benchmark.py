import pandas as pd
import time
import random
import os

from models import call_model_with_retry
from data_loader import get_benchmark_prompts
from model_config import MODEL_CATALOG
# Removed classify_prompt from imports as it is no longer used here
from main import initialize_market, run_simulation, judge_answer, print_leaderboard

# --- CONFIGURATION ---
TEST_SIZE = 100
TEST_DATA_OFFSET = 500
BENCHMARK_FILENAME = "benchmark_results_dec9_new_rewards.csv"
PRINT_SAVE_INTERVAL = 20

def run_baselines(prompts):
    """
    Runs the standard benchmark on all baseline models defined in configuration.
    Measures latency (excluding judge), cost (excluding judge), and quality score.
    """
    logs = []
    print(f"\n--- RUNNING BASELINES ({len(prompts)} Prompts) ---", flush=True)
    all_baseline_logs = []

    for model_id, specs in MODEL_CATALOG.items():
        
        if not specs.get("benchmark_model"): 
            continue
        
        current_model_logs = []

        print(f"\n🔵 Testing Baseline Model: {model_id}", flush=True)

        provider = specs['provider']
        throttle_sec = specs.get("throttle_delay", 0)

        for i, prompt_entry in enumerate(prompts):
            prompt_text = prompt_entry["text"]
            
            if i > 0 and throttle_sec > 0:
                time.sleep(throttle_sec)
            else:
                time.sleep(4)

            print(f"   [{i+1}/{len(prompts)}] Processing...", flush=True)
            
            agent_response = None
            latency = 0
            model_cost = 0
            score = None
            total_cost = 0

            # --- DIRECT CALL (No Classifier) ---
            agent_response, agent_latency = call_model_with_retry(provider, model_id, prompt_text)

            # Latency is purely the model generation time
            latency = agent_latency

            # 2. JUDGING (Only if response exists)
            if agent_response is not None:
                try:
                    # Calculate Real Cost
                    model_cost = (len(prompt_text) / 4 * specs["cost_in"]) + (len(agent_response) / 4 * specs["cost_out"])
                    total_cost = model_cost 
                    score, _ = judge_answer(prompt_text, agent_response)
                except Exception as e:
                    print(f"\n   ❌ Judge Failed prompt {i+1}: {e}", flush=True)
                    score = None

            print(f"   [{i+1}/{len(prompts)}] Baseline: {model_id:<25} | Score: {score if score else 0.0:.4f} | Latency: {latency:.4f}s", flush=True)

            log_entry = {
                "System": f"Baseline ({model_id})",
                "Winner": model_id,
                "Latency": latency,
                "Cost": total_cost,
                "Score": score
            }
            
            current_model_logs.append(log_entry)
            all_baseline_logs.append(log_entry)

            # 3. PERIODIC SAVE AND PRINT
            if (i + 1) % PRINT_SAVE_INTERVAL == 0:
                print(f"--- PERIODIC BENCHMARK SAVE at Round {i+1} for {model_id} ---", flush=True)
                
                pd.DataFrame(all_baseline_logs).to_csv(BENCHMARK_FILENAME, index=False)
                
                print(f"\n📈 Intermediate Leaderboard for {model_id} (Rounds 1 to {i+1}):", flush=True)
                temp_df = pd.DataFrame(current_model_logs) 
                
                if not temp_df.empty:
                    print(temp_df[["Cost", "Latency", "Score"]].mean())
                print("-" * 30, flush=True)


        print(f"   ✅ Completed {len(prompts)} prompts for {model_id}.", flush=True)
    return pd.DataFrame(all_baseline_logs)

def run_market(prompts):
    """
    Runs the full EV Router simulation in evaluation mode.
    """
    print(f"\n--- RUNNING EV ROUTER SIMULATION ({len(prompts)} Prompts) ---", flush=True)
    
    # Initialize the EV Router models (Cold start unless weights exist)
    market = initialize_market(load_weights_file=True)
    
    # run_simulation handles visibility when evaluate=True
    raw_results = run_simulation(market, prompts, evaluate=True)

    rows = []
    for _, result_row in raw_results.iterrows():
        total_reported_cost = result_row.get("compute_cost", 0)
        
        rows.append({
            "System": "Market", 
            "Winner": result_row["winner"],
            "Latency": result_row.get("latency", 0), 
            "Cost": total_reported_cost, 
            "Score": result_row["score"]
        })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    print(f"Loading {TEST_SIZE} prompts (Offset: {TEST_DATA_OFFSET})...", flush=True)
    prompts = get_benchmark_prompts(n_total=TEST_SIZE, start_index=TEST_DATA_OFFSET)

    if os.path.exists(BENCHMARK_FILENAME):
         print(f"Warning: Deleting existing file {BENCHMARK_FILENAME} to ensure a clean benchmark run.", flush=True)
         os.remove(BENCHMARK_FILENAME)
         
    df_base = run_baselines(prompts)
    df_market = run_market(prompts)

    combined = pd.concat([df_base, df_market])
    
    combined.to_csv(BENCHMARK_FILENAME, index=False)

    print("\n" + "="*80)
    print("🏁 FINAL BENCHMARK REPORT")
    print("="*80)
    
    if not combined.empty:
        print(combined.groupby("System")[["Cost", "Latency", "Score"]].mean())
    
    print("="*80)
    print(f"Detailed results saved to {BENCHMARK_FILENAME}")