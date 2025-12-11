import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import beta as beta_dist
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns  # <--- ADDED for EV plots
import json
import sys
import os

# --- CONFIGURATION ---
SIGNIFICANCE_LEVEL = 0.05
CONFIDENCE_DELTA = 0.05
benchmark_file = "benchmark_results_dec_8.csv"
MARKET_FILE = "market_weights_dec_8.json"

# --- PLOT NAMES ---
PLOT_CONFIDENCE = "score_confidence_plot_clean_250_dec8.png"
PLOT_COST_LATENCY = "cost_vs_latency_scatter_clean_250_dec8.png"
PLOT_PERFORMANCE = "cumulative_progress_plot_clean_250_dec8.png"
PLOT_PARETO = "pareto_efficiency_plot_250_dec8.png"
PLOT_EV_DIST = "expected_value_distribution_plot.png"

# --- ADDED CONSTANTS FOR EV MATH ---
CATEGORIES = {
    "easy": { "output_est": 50, "reward": 0.02 },
    "hard": { "output_est": 500, "reward": 0.15 }
}

# Simplified cost lookup for visualization (since JSON doesn't store specs)
# Defaulting to Gemini Flash prices ($0.10/$0.40) as baseline if unknown
DEFAULT_COST_IN = 0.10 / 1e6
DEFAULT_COST_OUT = 0.40 / 1e6

# --- STATISTICAL FUNCTIONS ---

def calculate_hoeffding_epsilon(N, delta):
    if N <= 0: return 0
    return np.sqrt(np.log(2 / delta) / (2 * N))

def calculate_min_hoeffding_samples(delta, margin):
    """Calculates N required to detect a specific margin with 1-delta confidence."""
    if margin <= 0: return float('inf')
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

    required_margin = 0.05
    n_required = calculate_min_hoeffding_samples(CONFIDENCE_DELTA, required_margin)
    print(f"📊 Samples required to detect {required_margin} margin (95% CI): {int(n_required)}")

# --- VISUALIZATION FUNCTIONS ---

def set_plot_style():
    """Safely sets a clean plot style compatible with different Matplotlib versions."""
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        plt.style.use('seaborn-whitegrid')

def create_confidence_plot(df: pd.DataFrame):
    set_plot_style()
    grouped = df.groupby('System')['Score'].agg(['mean', 'count'])
    grouped['epsilon'] = grouped['count'].apply(lambda n: calculate_hoeffding_epsilon(n, CONFIDENCE_DELTA))
    
    systems = grouped.index
    means = grouped['mean'].values
    errors = grouped['epsilon'].values 

    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#bdc3c7'] * len(systems)
    if 'Market' in systems:
        colors[systems.get_loc('Market')] = '#2980b9' 

    y_pos = np.arange(len(systems))
    ax.barh(y_pos, means, xerr=errors, capsize=5, color=colors, alpha=0.9)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(systems, fontsize=10)
    ax.set_xlabel("Average Score")
    ax.set_xlim(0, 1.1)
    ax.set_title(f"Score Comparison with {1 - CONFIDENCE_DELTA:.0%} Hoeffding CI")
    
    plt.tight_layout()
    plt.savefig(PLOT_CONFIDENCE)
    plt.close()
    print(f"📈 Saved: {PLOT_CONFIDENCE}")

def create_cost_vs_latency_plot(df: pd.DataFrame):
    set_plot_style()
    plt.figure(figsize=(10, 7))
    systems = df['System'].unique()
    cmap = plt.get_cmap('Dark2') 
    
    for i, system in enumerate(systems):
        subset = df[df['System'] == system]
        plt.scatter(
            subset['Latency'], 
            subset['Cost'] * 1000, 
            label=system, 
            alpha=0.7, 
            edgecolors='white',
            color=cmap(i % 8) 
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
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    cmap = plt.get_cmap('Dark2')
    
    for i, system in enumerate(systems):
        subset = df[df['System'] == system].reset_index(drop=True)
        color = cmap(i % 8)
        ax1.plot(subset.index + 1, subset['Cost'].cumsum(), label=system, color=color, linewidth=2)
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
    
    summary = df.groupby('System').agg({
        'Cost': 'mean',
        'Score': 'mean',
        'Latency': 'mean'
    }).reset_index()

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

    for i, row in summary.iterrows():
        label = row['System'].replace('Baseline (', '').replace(')', '')
        weight = 'bold' if 'Market' in label else 'normal'
        size = 12 if 'Market' in label else 9
        plt.text(row['Cost'] * 1.05, row['Score'], label, fontsize=size, fontweight=weight, verticalalignment='center')

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

def create_beta_distribution_plot(market_weights: dict):
    set_plot_style()
    model_data = []
    for model_id, data in market_weights.items():
        skills = data.get("skills", {})
        for difficulty, params in skills.items():
            model_data.append({
                'model_id': model_id,
                'difficulty': difficulty,
                'alpha': params['alpha'],
                'beta': params['beta']
            })
    
    df_beta = pd.DataFrame(model_data)
    if df_beta.empty: return

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    plt.subplots_adjust(wspace=0.05)
    
    x = np.linspace(0.01, 0.99, 100)
    models = df_beta['model_id'].unique()
    cmap = plt.get_cmap('viridis', len(models))
    color_map = {model: cmap(i) for i, model in enumerate(models)}
    
    ax = axes[0]
    ax.set_title('Easy Prompts (Predicted Quality Score)', fontsize=14)
    ax.set_xlabel('Score (0 to 1)', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_xlim(0, 1)
    
    for i, row in df_beta[df_beta['difficulty'] == 'easy'].iterrows():
        pdf = beta_dist.pdf(x, row['alpha'], row['beta'])
        ax.plot(x, pdf, label=row['model_id'], color=color_map[row['model_id']])
    ax.legend(loc='upper left', fontsize=8)

    ax = axes[1]
    ax.set_title('Hard Prompts (Predicted Quality Score)', fontsize=14)
    ax.set_xlabel('Score (0 to 1)', fontsize=12)
    ax.set_xlim(0, 1)
    
    for i, row in df_beta[df_beta['difficulty'] == 'hard'].iterrows():
        pdf = beta_dist.pdf(x, row['alpha'], row['beta'])
        ax.plot(x, pdf, label=row['model_id'], color=color_map[row['model_id']])

    fig.suptitle('Beta Distribution of Expected Quality by Difficulty', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig("beta_distribution_plot.png")
    plt.close()
    print("📈 Saved: beta_distribution_plot.png")

# --- NEW: EXPECTED VALUE DISTRIBUTION PLOT ---

def plot_ev_distributions(market_weights: dict, n_samples=10000):
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    AVG_INPUT_LEN = 500  # chars
    
    for i, complexity in enumerate(["easy", "hard"]):
        ax = axes[i]
        reward = CATEGORIES[complexity]["reward"]
        output_est = CATEGORIES[complexity]["output_est"]
        
        for mid, data in market_weights.items():
            stats = data["skills"].get(complexity, data["skills"].get("hard"))
            alpha = stats["alpha"]
            beta = stats["beta"]
            ema_latency = data.get("ema_latency", 0.5)
            
            # 1. Sample from Quality Belief
            quality_samples = np.random.beta(alpha, beta, n_samples)
            
            # 2. Calculate Costs
            # Note: Using default costs since they aren't in the JSON weights
            compute_cost = (AVG_INPUT_LEN/4 * DEFAULT_COST_IN) + \
                           (output_est * DEFAULT_COST_OUT)
            time_cost = ema_latency * 0.002 # Time Penalty
            total_cost = compute_cost + time_cost
            
            # 3. Transform to PROFIT
            profit_samples = (quality_samples * reward) - total_cost
            
            # 4. Plot
            sns.kdeplot(profit_samples, ax=ax, label=f"{mid.split('/')[-1]}", fill=True, alpha=0.1)

        ax.set_title(f"Expected Profit Distribution ({complexity.upper()})")
        ax.set_xlabel("Net Profit ($)")
        ax.set_ylabel("Probability Density")
        ax.axvline(0, color='red', linestyle='--', alpha=0.5, label="Break Even ($0)")
        ax.legend(loc='upper left', fontsize='small')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_EV_DIST)
    plt.close()
    print(f"📈 Saved: {PLOT_EV_DIST}")

# --- MAIN ---

if __name__ == "__main__":
    try:
        with open(MARKET_FILE, 'r') as f:
            market_weights_data = json.load(f)
            
        print("\ngenerating Beta distribution plot...")
        create_beta_distribution_plot(market_weights_data)

        # NEW: Generate EV Plot
        print("\ngenerating Expected Value distribution plot...")
        plot_ev_distributions(market_weights_data)

    except FileNotFoundError:
        print(f"⚠️ Warning: {MARKET_FILE} not found. Skipping weight plots.")
    except json.JSONDecodeError:
        print(f"❌ Error: Could not decode {MARKET_FILE}. Check file format.")

    # Generate benchmark plots
    # if not os.path.exists(benchmark_file):
    #     print(f"Error: {benchmark_file} not found.")
    #     sys.exit(1)
        
    # full_df = pd.read_csv(benchmark_file)

    # market_df = full_df[full_df['System'] == 'Market']
    # market_size = len(market_df)

    # # to solve sample mismatches due to failures etc.
    # df = full_df.groupby('System').head(market_size).reset_index(drop=True)
    # df['Score'] = df['Score'].fillna(0.0)
    
    # baselines = [s for s in df['System'].unique() if s.startswith('Baseline')]
    # market_system = "Market"

    # if not baselines:
    #     print("No Baselines found.")
    # else:
    #     for baseline in baselines:
    #         run_statistical_analysis(df, baseline, market_system)

    # print("\ngenerating plots...")
    # create_confidence_plot(df)
    # create_cost_vs_latency_plot(df)
    # create_cumulative_performance_plot(df)
    # create_pareto_frontier_plot(df)