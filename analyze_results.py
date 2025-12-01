import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import sys
import os

# --- CONFIGURATION ---
SIGNIFICANCE_LEVEL = 0.05  # Alpha for t-test
CONFIDENCE_DELTA = 0.05    # Delta for Hoeffding (1 - delta = 95% confidence)

# --- STATISTICAL FUNCTIONS ---

def calculate_hoeffding_epsilon(N, delta):
    """
    Calculates the Hoeffding bound (epsilon) for a given sample size (N) and confidence risk (delta).
    Assumes score range (b-a) = 1 (since scores are 0-1).
    """
    if N <= 0:
        return 0
    # Hoeffding Margin of Error: epsilon = sqrt( ln(2/delta) / (2 * N) )
    epsilon = np.sqrt(np.log(2 / delta) / (2 * N))
    return epsilon

def run_statistical_analysis(df: pd.DataFrame, baseline_system: str, market_system: str = "Market"):
    """
    Runs Hoeffding and Paired T-Test comparing Market to a specified Baseline.
    """
    print(f"\n🔬 Running Statistical Analysis: {market_system} vs. {baseline_system}")
    
    # 1. Prepare Data
    # Get the raw list of scores (0s and 1s) for each system
    df_baseline = df[df['System'] == baseline_system].reset_index()
    df_market = df[df['System'] == market_system].reset_index()

    N = len(df_market)
    if N == 0 or len(df_baseline) != N:
        print("❌ Error: Cannot run analysis. Sample sizes do not match or are zero.")
        return

    market_scores = df_market['Score'].values
    baseline_scores = df_baseline['Score'].values
    
    market_mean = market_scores.mean()
    baseline_mean = baseline_scores.mean()
    
    print(f"Sample Size (N): {N}")
    print(f"Market Mean Score: {market_mean:.6f}")
    print(f"{baseline_system} Mean Score: {baseline_mean:.6f}")
    print("-" * 50)
    
    # --- A. Hoeffding's Inequality ---
    epsilon = calculate_hoeffding_epsilon(N, CONFIDENCE_DELTA)
    
    print(f"Hoeffding's Inequality (Confidence: {1 - CONFIDENCE_DELTA:.0%} / Risk: {CONFIDENCE_DELTA})")
    print(f"Tolerance (epsilon): +/- {epsilon:.4f}")
    
    # Market Confidence Interval (CI)
    market_ci_low = market_mean - epsilon
    market_ci_high = min(1.0, market_mean + epsilon) # Cap CI at 1.0 since score can't exceed 1
    print(f"True Market Score is in [{market_ci_low:.4f}, {market_ci_high:.4f}]")

    # Conclusion check
    is_better_confidently = market_ci_low > baseline_mean
    if is_better_confidently:
        print(f"✅ Hoeffding: Market is CONFIDENTLY better than Baseline because the entire CI is above the baseline mean.")
    elif market_mean > baseline_mean:
        print(f"⚠️ Hoeffding: Market is numerically better, but the CI overlaps with the baseline mean.")
    else:
        print(f"❌ Hoeffding: Market is numerically worse or statistically tied with the baseline.")

    print("-" * 50)

    # --- B. Paired T-Test ---
    t_statistic, p_two_tailed = stats.ttest_rel(market_scores, baseline_scores)
    
    # One-tailed p-value for H_a: mean(Market) > mean(Baseline)
    if t_statistic > 0:
        p_one_tailed = p_two_tailed / 2
    else:
        # If the T-statistic is negative or zero, we cannot claim Market is better, 
        # so the one-tailed p-value remains high (0.5 or greater).
        p_one_tailed = 1 - (p_two_tailed / 2) 

    print("Paired T-Test (One-Tailed: Market > Baseline)")
    print(f"T-Statistic: {t_statistic:.4f}")
    print(f"P-Value (One-Tailed): {p_one_tailed:.4f}")
    
    is_significant = p_one_tailed < SIGNIFICANCE_LEVEL
    if is_significant:
        print(f"✅ T-Test: Difference is STATISTICALLY SIGNIFICANT at the {SIGNIFICANCE_LEVEL} level.")
    else:
        print(f"❌ T-Test: Difference is NOT statistically significant at the {SIGNIFICANCE_LEVEL} level.")
        
    return market_mean, baseline_mean, epsilon, is_better_confidently, is_significant

# --- VISUALIZATION FUNCTIONS (Nicer Plots) ---

def create_confidence_plot(df: pd.DataFrame):
    """Visualizes the mean scores and Hoeffding confidence intervals."""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    grouped = df.groupby('System')['Score'].agg(['mean', 'count'])
    grouped['N'] = grouped['count']
    grouped['epsilon'] = grouped['N'].apply(lambda n: calculate_hoeffding_epsilon(n, CONFIDENCE_DELTA))
    grouped['ci_low'] = grouped['mean'] - grouped['epsilon']
    
    systems = grouped.index
    means = grouped['mean'].values
    errors = grouped['epsilon'].values 

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Assign specific colors, making 'Market' stand out
    colors = ['gray'] * len(systems)
    if 'Market' in systems:
        market_index = systems.get_loc('Market')
        colors[market_index] = 'royalblue'

    ax.bar(systems, means, yerr=errors, capsize=6, color=colors, alpha=0.8)
    
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Average Score")
    ax.set_title(f"Score Comparison with {1 - CONFIDENCE_DELTA:.0%} Hoeffding Confidence Intervals")
    
    # Add labels for mean
    for i, system in enumerate(systems):
        mean_val = means[i]
        ax.text(i, mean_val + 0.03, f"{mean_val:.3f}", ha='center', color='black', fontsize=10, fontweight='bold')

    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig("score_confidence_plot_clean.png")
    print("📈 Visualization saved to 'score_confidence_plot_clean.png'")
    

def create_cost_vs_latency_plot(df: pd.DataFrame):
    """Visualizes the Cost vs. Latency for every run across all systems."""
    plt.style.use('seaborn-v0_8-deep') 
    plt.figure(figsize=(10, 7))
    
    systems = df['System'].unique()
    
    # Use a color map for distinct colors
    cmap = plt.cm.get_cmap('Dark2', len(systems))
    
    for i, system in enumerate(systems):
        subset = df[df['System'] == system]
        plt.scatter(
            subset['Latency'], 
            subset['Cost'] * 1000, # Convert USD to mUSD (Millicent) for readability
            label=system, 
            alpha=0.7,
            s=80, 
            edgecolors='k', 
            linewidths=0.5,
            color=cmap(i)
        )

    plt.xscale('log') 
    plt.yscale('log') 
    plt.xlabel("Latency per Prompt (Seconds, Log Scale)")
    plt.ylabel("Cost per Prompt (mUSD - Millicent, Log Scale)")
    plt.title("Trade-off: Latency vs. Cost per Execution (Log-Log Scale)")
    plt.legend(loc='upper left', title="System", fontsize=9)
    plt.grid(True, which="both", ls="--", linewidth=0.3, alpha=0.5)
    plt.tight_layout()
    plt.savefig("cost_vs_latency_scatter_clean.png")
    print("📈 Visualization saved to 'cost_vs_latency_scatter_clean.png'")
    

def create_cumulative_performance_plot(df: pd.DataFrame):
    """Visualizes cumulative cost and running average score per system."""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    systems = df['System'].unique()
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # --- Y-Axis 1: Cumulative Cost (Blue tones) ---
    ax1.set_xlabel("Prompt Index (Test Progress)")
    ax1.set_ylabel("Cumulative Cost (USD)", color='#0066cc', fontweight='bold')
    
    # --- Y-Axis 2: Running Average Score (Red/Orange tones) ---
    ax2 = ax1.twinx()
    ax2.set_ylabel("Running Average Score", color='#ff6600', fontweight='bold')
    ax2.set_ylim(0, 1.05) 
    
    cmap = plt.cm.get_cmap('Dark2', len(systems))
    
    # Store lines/labels for a unified legend
    all_lines = []
    all_labels = []

    for i, system in enumerate(systems):
        subset = df[df['System'] == system].copy().reset_index(drop=True)
        subset['Cumulative_Cost'] = subset['Cost'].cumsum()
        subset['Running_Score'] = subset['Score'].expanding().mean()
        
        # Plot Cost (Solid line)
        line1, = ax1.plot(
            subset.index + 1, 
            subset['Cumulative_Cost'], 
            label=f"{system} Cost", 
            color=cmap(i), 
            linestyle='-', 
            linewidth=2
        )
        # Plot Score (Dashed line)
        line2, = ax2.plot(
            subset.index + 1, 
            subset['Running_Score'], 
            label=f"{system} Score", 
            color=cmap(i), 
            linestyle='--', 
            linewidth=2
        )
        all_lines.extend([line1, line2])
        all_labels.extend([f'{system} Cost', f'{system} Score'])


    ax1.tick_params(axis='y', labelcolor='#0066cc')
    ax2.tick_params(axis='y', labelcolor='#ff6600')
    
    ax1.legend(all_lines, all_labels, loc='center left', bbox_to_anchor=(1.05, 0.5), title="System Metric")
    
    plt.title("Cumulative Benchmark Progress: Cost & Running Average Score")
    plt.tight_layout()
    plt.savefig("cumulative_progress_plot_clean.png")
    print("📈 Visualization saved to 'cumulative_progress_plot_clean.png'")
    

# --- MAIN EXECUTION BLOCK ---

if __name__ == "__main__":
    try:
        # Check if the results file exists
        if not os.path.exists("benchmark_results.csv"):
            print("Error: 'benchmark_results.csv' not found. Please run the main benchmark script first to generate the data.")
            sys.exit(1)
            
        df = pd.read_csv("benchmark_results.csv")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
        
    # --- 1. Statistical Analysis ---
    
    # Identify all baselines
    baselines = [s for s in df['System'].unique() if s.startswith('Baseline')]
    market_system = "Market"
    
    if baselines:
        # Compare Market against the lowest performing baseline (to make the strongest non-superiority claim)
        baseline_scores = df[df['System'].str.startswith('Baseline')].groupby('System')['Score'].mean()
        comparison_baseline = baseline_scores.idxmin()
        run_statistical_analysis(df, comparison_baseline, market_system)
    else:
        print("No Baselines found in the data to compare against.")

    # --- 2. Visualization ---

    # Create the visualization for ALL systems
    create_confidence_plot(df)
    create_cost_vs_latency_plot(df)
    create_cumulative_performance_plot(df)