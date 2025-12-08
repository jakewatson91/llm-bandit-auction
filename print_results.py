import pandas as pd
import os

BENCHMARK_FILE = "benchmark_results_200.csv"

def pct_diff(x, baseline):
    if baseline == 0:
        return "N/A"
    return f"{((x - baseline) / baseline) * 100:.1f}%"

if not os.path.exists(BENCHMARK_FILE):
    print(f"❌ Error: benchmark file '{BENCHMARK_FILE}' not found.")
    exit(1)

df = pd.read_csv(BENCHMARK_FILE)

# Clean NaNs
df['Score'] = df['Score'].fillna(0.0)
df['Cost'] = df['Cost'].fillna(0.0)
df['Latency'] = df['Latency'].fillna(0.0)

# Build summary table
summary = (
    df.groupby("System")
      .agg(
          Avg_Score=("Score", "mean"),
          Avg_Cost=("Cost", "mean"),
          Avg_Latency=("Latency", "mean"),
          N=("Score", "count")
      )
      .reset_index()
)

# Extract Market as reference
market_row = summary[summary["System"] == "Market"]
if market_row.empty:
    print("❌ No Market system found in summary.")
    exit(1)

market_score = market_row["Avg_Score"].values[0]
market_cost = market_row["Avg_Cost"].values[0]
market_latency = market_row["Avg_Latency"].values[0]

# Compute differences
summary["Score Δ% vs Market"] = summary["Avg_Score"].apply(lambda x: pct_diff(x, market_score))
summary["Cost Δ% vs Market"] = summary["Avg_Cost"].apply(lambda x: pct_diff(x, market_cost))
summary["Latency Δ% vs Market"] = summary["Avg_Latency"].apply(lambda x: pct_diff(x, market_latency))

# Pretty formatting
summary["Avg_Score"] = summary["Avg_Score"].map(lambda x: f"{x:.4f}")
summary["Avg_Cost"] = summary["Avg_Cost"].map(lambda x: f"${x:.6f}")
summary["Avg_Latency"] = summary["Avg_Latency"].map(lambda x: f"{x:.4f}s")

print("\n==================== SUMMARY TABLE WITH % DIFFERENCES ====================\n")
print(summary.to_string(index=False))
print("\n==========================================================================\n")
