import pandas as pd
import time
import os

# Import your existing modules
from data_loader import get_benchmark_prompts
from main import initialize_market, run_simulation

# --- CONFIGURATION ---
TEST_SIZE = 100
TEST_DATA_OFFSET = 500

# Path to the PREVIOUS benchmark file containing the baselines you want to keep
EXISTING_BENCHMARK_PATH = "/Users/jakewatson/Desktop/cs525-2025/Project/benchmark_results_200.csv"

# Path for the NEW combined output
NEW_BENCHMARK_FILENAME = "benchmark_results_dec_5.csv"

def run_market(prompts):
    """
    Runs the full market simulation in evaluation mode using the main simulation logic.
    Extracts the 'winner' and performance metrics for comparison against baselines.
    """
    print(f"\n--- RUNNING MARKET SIMULATION ({len(prompts)} Prompts) ---", flush=True)
    
    # Initialize market with saved weights
    market = initialize_market(load_weights_file=True)
    
    # Run simulation (evaluate=True usually hides verbose internal prints)
    raw_results = run_simulation(market, prompts, evaluate=True)

    rows = []
    for _, result_row in raw_results.iterrows():
        # Ensure we capture cost, handling cases where it might be missing
        total_reported_cost = result_row.get("compute_cost", 0)
        
        rows.append({
            "System": "Market", 
            "Winner": result_row["winner"],
            "Latency": result_row.get("latency", 0), 
            "Cost": total_reported_cost, 
            "Score": result_row["score"]
        })
    
    df = pd.DataFrame(rows)
    print(f"   ✅ Market run complete. Generated {len(df)} rows.")
    return df

def load_existing_baselines(filepath, limit=None):
    """
    Loads an existing benchmark CSV and filters out old Market runs,
    returning only the Baseline rows. 
    Optionally limits the number of rows per baseline to match the current run.
    """
    if not os.path.exists(filepath):
        print(f"⚠️  WARNING: Could not find existing file at {filepath}")
        print("    No baselines will be included.")
        return pd.DataFrame()

    print(f"📂 Loading existing results from: {filepath}")
    df = pd.read_csv(filepath)
    
    # Filter: Keep rows where System is NOT "Market"
    # This keeps "Baseline (Llama-3-8b)", "Baseline (GPT-4o)", etc.
    df_baselines = df[df["System"] != "Market"].copy()
    
    if limit is not None:
        print(f"   ✂️  Limiting baselines to {limit} samples per system to match current Market run...")
        df_baselines = df_baselines.groupby("System").head(limit).reset_index(drop=True)

    print(f"   Found {len(df)} total rows in file.")
    print(f"   Extracted {len(df_baselines)} baseline rows to preserve.")
    
    return df_baselines

if __name__ == "__main__":
    # 1. Load the new prompts to test the Market on
    print(f"Loading {TEST_SIZE} prompts (Offset: {TEST_DATA_OFFSET})...", flush=True)
    prompts = get_benchmark_prompts(n_total=TEST_SIZE, start_index=TEST_DATA_OFFSET)

    # 2. Run ONLY the Market Simulation
    df_market = run_market(prompts)
    
    # Calculate how many samples we actually got (in case of failures or partial runs)
    current_sample_count = len(df_market)

    # 3. Load the OLD Baselines, limiting them to the number of market samples
    df_base = load_existing_baselines(EXISTING_BENCHMARK_PATH, limit=current_sample_count)

    # 4. Combine them
    combined = pd.concat([df_base, df_market], ignore_index=True)
    
    # 5. Save
    combined.to_csv(NEW_BENCHMARK_FILENAME, index=False)

    print("\n" + "="*80)
    print("🏁 FINAL MIXED BENCHMARK REPORT")
    print("="*80)
    
    if not combined.empty:
        # Group by System to see how Market compares to the loaded Baselines
        print(combined.groupby("System")[["Cost", "Latency", "Score"]].mean())
    
    print("="*80)
    print(f"Detailed results (Old Baselines + New Market) saved to {NEW_BENCHMARK_FILENAME}")