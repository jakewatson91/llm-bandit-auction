import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
import os

# --- CONFIGURATION ---
SIGNIFICANCE_LEVEL = 0.05
CONFIDENCE_DELTA = 0.05
benchmark_file = "benchmark_results_dec_5.csv"

# --- PLOT NAMES ---
PLOT_CONFIDENCE = "score_confidence_plot_clean_200.png"
PLOT_COST_LATENCY = "cost_vs_latency_scatter_clean_200.png"
PLOT_PERFORMANCE = "cumulative_progress_plot_clean_200.png"
PLOT_PARETO = "pareto_efficiency_plot_200.png"

# --- STATISTICAL FUNCTIONS ---

def calculate_hoeffding_epsilon(N, delta):
    if N <= 0: return 0
    return np.sqrt(np.log(2 / delta) / (2 * N))

def calculate_min_hoeffding_samples(delta, margin):
    """Calculates N required to detect a specific margin with 1-delta confidence."""
    if margin <= 0: return float('inf')
    # N = ln(2 / delta) / (2 * margin^2)
    numerator = np.log(2 / delta)
    denominator = 2 * (margin ** 2)
    return np.ceil(numerator / denominator)

def run_statistical_analysis(df: pd.DataFrame, baseline_system: str, market_system: str = "Market"):
    print(f"\n🔬 Running Statistical Analysis: {market_system} vs. {baseline_system}")
    
    df_baseline = df[df['System'] == baseline_system]
    df_market = df[df['System'] == market_system]

    N = len(df_market)
    if N == 0 or len(df_baseline) != N:
        print(f"❌ Error: Sample sizes mismatch (Market: {N}, Baseline: {len(df_baseline)}). Skipping.")
        return

    market_scores = df_market['Score'].values
    baseline_scores = df_baseline['Score'].values
    
    market_mean = market_scores.mean()
    baseline_mean = baseline_scores.mean()
    
    print(f"Sample Size (N): {N}")
    print(f"Market Mean: {market_mean:.4f} | {baseline_system} Mean: {baseline_mean:.4f}")
    
    # --- HOEFFDING TEST ---
    epsilon = calculate_hoeffding_epsilon(N, CONFIDENCE_DELTA)
    market_ci_low = market_mean - epsilon
    
    if market_ci_low > baseline_mean:
        print(f"✅ Hoeffding: Market is CONFIDENTLY better (Lower CI {market_ci_low:.4f} > Baseline Mean).")
    else:
        print(f"❌ Hoeffding: Market is NOT confidently better (CI overlaps or falls below baseline).")

    # --- T-TEST ---
    t_stat, p_two = stats.ttest_rel(market_scores, baseline_scores)
    p_one = p_two / 2 if t_stat > 0 else 1.0 - (p_two / 2)

    if p_one < SIGNIFICANCE_LEVEL:
        print(f"✅ T-Test: Difference is SIGNIFICANT (p={p_one:.4f}).")
    else:
        print(f"❌ T-Test: Difference is NOT significant (p={p_one:.4f}).")

    # --- REQUIRED SAMPLE SIZE ---
    required_margin = 0.05
    n_required = calculate_min_hoeffding_samples(CONFIDENCE_DELTA, required_margin)
    print(f"📊 Samples required to detect {required_margin} margin (95% CI): {int(n_required)}")

# --- VISUALIZATION FUNCTIONS ---

def set_plot_style():
    """Safely sets a clean plot style compatible with different Matplotlib versions."""
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        # Fallback for older matplotlib versions
        plt.style.use('seaborn-whitegrid')

def create_confidence_plot(df: pd.DataFrame):
    set_plot_style()
    
    grouped = df.groupby('System')['Score'].agg(['mean', 'count'])
    grouped['epsilon'] = grouped['count'].apply(lambda n: calculate_hoeffding_epsilon(n, CONFIDENCE_DELTA))
    
    systems = grouped.index
    means = grouped['mean'].values
    errors = grouped['epsilon'].values 

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color logic
    colors = ['#bdc3c7'] * len(systems) # Default Gray
    if 'Market' in systems:
        colors[systems.get_loc('Market')] = '#2980b9' # Blue

    # Use Horizontal bars (barh) to prevent label overlap
    y_pos = np.arange(len(systems))
    ax.barh(y_pos, means, xerr=errors, capsize=5, color=colors, alpha=0.9)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(systems, fontsize=10)
    ax.set_xlabel("Average Score")
    ax.set_xlim(0, 1.1)
    ax.set_title(f"Score Comparison with {1 - CONFIDENCE_DELTA:.0%} Hoeffding CI")
    
    plt.tight_layout()
    plt.savefig(PLOT_CONFIDENCE)
    plt.close() # Close memory
    print(f"📈 Saved: {PLOT_CONFIDENCE}")

def create_cost_vs_latency_plot(df: pd.DataFrame):
    set_plot_style()
    plt.figure(figsize=(10, 7))
    
    systems = df['System'].unique()
    
    # Safe colormap retrieval
    cmap = plt.get_cmap('Dark2') 
    
    for i, system in enumerate(systems):
        subset = df[df['System'] == system]
        plt.scatter(
            subset['Latency'], 
            subset['Cost'] * 1000, 
            label=system, 
            alpha=0.7, 
            edgecolors='white',
            color=cmap(i % 8) # Modulo to prevent index error
        )

    plt.xscale('log') 
    plt.yscale('log') 
    plt.xlabel("Latency (s) [Log]")
    plt.ylabel("Cost (mUSD) [Log]")
    plt.title("Latency vs. Cost per Prompt")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_COST_LATENCY)
    plt.close()
    print(f"📈 Saved: {PLOT_COST_LATENCY}")

def create_cumulative_performance_plot(df: pd.DataFrame):
    set_plot_style()
    systems = df['System'].unique()
    
    # Use 2 rows instead of twin-axis for better readability if "broken"
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    cmap = plt.get_cmap('Dark2')
    
    for i, system in enumerate(systems):
        subset = df[df['System'] == system].reset_index(drop=True)
        color = cmap(i % 8)
        
        # Plot Cost
        ax1.plot(subset.index + 1, subset['Cost'].cumsum(), label=system, color=color, linewidth=2)
        
        # Plot Score
        ax2.plot(subset.index + 1, subset['Score'].expanding().mean(), label=system, color=color, linewidth=2, linestyle='--')

    ax1.set_ylabel("Cumulative Cost ($)")
    ax1.legend(loc='upper left')
    ax1.set_title("Cumulative Cost")
    
    ax2.set_ylabel("Running Avg Score")
    ax2.set_xlabel("Prompt Index")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Performance Stability")
    
    plt.tight_layout()
    plt.savefig(PLOT_PERFORMANCE)
    plt.close()
    print(f"📈 Saved: {PLOT_PERFORMANCE}")

def create_pareto_frontier_plot(df: pd.DataFrame):
    set_plot_style()
    plt.figure(figsize=(12, 8))
    
    # 1. Aggregate Data
    summary = df.groupby('System').agg({
        'Cost': 'mean',
        'Score': 'mean',
        'Latency': 'mean'
    }).reset_index()

    # 2. Identify the Pareto Frontier
    sorted_data = summary.sort_values('Cost')
    frontier_costs = []
    frontier_scores = []
    current_max_score = -1.0
    
    for _, row in sorted_data.iterrows():
        if row['Score'] > current_max_score:
            frontier_costs.append(row['Cost'])
            frontier_scores.append(row['Score'])
            current_max_score = row['Score']
            
    plt.plot(frontier_costs, frontier_scores, 'k--', alpha=0.3, label='Pareto Frontier')

    # 3. Create Scatter Plot
    sc = plt.scatter(
        summary['Cost'], 
        summary['Score'], 
        c=summary['Latency'], 
        cmap='RdYlGn_r', 
        s=150, 
        edgecolors='black',
        alpha=0.9,
        zorder=10
    )

    # 4. Annotate Points
    for i, row in summary.iterrows():
        label = row['System'].replace('Baseline (', '').replace(')', '')
        weight = 'bold' if 'Market' in label else 'normal'
        size = 12 if 'Market' in label else 9
        plt.text(row['Cost'] * 1.05, row['Score'], label, fontsize=size, fontweight=weight, verticalalignment='center')

    # 5. Formatting
    plt.xscale('log')
    import matplotlib.ticker as mticker
    plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: '${:g}'.format(y)))
    
    plt.xlabel('Average Cost ($) [Log Scale]', fontsize=12)
    plt.ylabel('Average Score (0-1)', fontsize=12)
    plt.title('Market Efficiency: Score vs. Cost (Color = Latency)', fontsize=14)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    cbar = plt.colorbar(sc)
    cbar.set_label('Avg Latency (seconds)', rotation=270, labelpad=15)

    plt.tight_layout()
    plt.savefig(PLOT_PARETO)
    plt.close()
    print(f"📈 Saved: {PLOT_PARETO}")

# --- MAIN ---

if __name__ == "__main__":
    if not os.path.exists(benchmark_file):
        print(f"Error: {benchmark_file} not found.")
        sys.exit(1)
        
    full_df = pd.read_csv(benchmark_file)

    market_df = full_df[full_df['System'] == 'Market']
    market_size = len(market_df)

    # to solve sample mismatches due to failures etc.
    df = full_df.groupby('System').head(market_size).reset_index(drop=True)
    # Replace all NaN scores with 0.0
    df['Score'] = df['Score'].fillna(0.0)
    
    baselines = [s for s in df['System'].unique() if s.startswith('Baseline')]
    market_system = "Market"

    if not baselines:
        print("No Baselines found.")
    else:
        for baseline in baselines:
            run_statistical_analysis(df, baseline, market_system)

    # 2. Run Visualizations
    print("\ngenerating plots...")
    create_confidence_plot(df)
    create_cost_vs_latency_plot(df)
    create_cumulative_performance_plot(df)
    create_pareto_frontier_plot(df)