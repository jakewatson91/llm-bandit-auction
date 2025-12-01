import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_plots(csv_file="benchmark_results.csv"):
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Run benchmark.py first.")
        return

    df = pd.read_csv(csv_file)
    sns.set_theme(style="whitegrid")
    
    # 1. HEAD-TO-HEAD COMPARISON (Bar Chart)
    print("Generating System Comparison...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    sns.barplot(data=df, x="System", y="Cost", ax=axes[0], palette="viridis", errorbar=None)
    axes[0].set_title("Average Cost ($) - Lower is Better")
    axes[0].set_ylabel("Cost per Query")
    axes[0].tick_params(axis='x', rotation=45)
    
    sns.barplot(data=df, x="System", y="Latency", ax=axes[1], palette="magma", errorbar=None)
    axes[1].set_title("Average Latency (s) - Lower is Better")
    axes[1].set_ylabel("Seconds")
    axes[1].tick_params(axis='x', rotation=45)
    
    sns.barplot(data=df, x="System", y="Score", ax=axes[2], palette="rocket", errorbar=None)
    axes[2].set_title("Average Quality (0-1) - Higher is Better")
    axes[2].set_ylim(0, 1.1)
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig("system_comparison.png")
    print(" > Saved system_comparison.png")
    
    # 2. MARKET ROUTING (Pie Chart)
    print("Generating Routing Distribution...")
    market_df = df[df['System'] == "Market (Ours)"]
    
    if not market_df.empty and 'Winner' in market_df.columns:
        plt.figure(figsize=(10, 6))
        winner_counts = market_df['Winner'].value_counts()
        
        plt.pie(winner_counts, labels=winner_counts.index, autopct='%1.1f%%', 
                colors=sns.color_palette("pastel"), startangle=140)
        plt.title("Which Agents Won in the Market?")
        plt.savefig("market_routing.png")
        print(" > Saved market_routing.png")

    # 3. CATEGORY COST BREAKDOWN (Grouped Bar Chart)
    print("Generating Category Breakdown...")
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="Category", y="Cost", hue="System", palette="muted", errorbar=None)
    plt.title("Cost Efficiency by Category")
    plt.ylabel("Cost ($)")
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("category_cost.png")
    print(" > Saved category_cost.png")

if __name__ == "__main__":
    generate_plots()