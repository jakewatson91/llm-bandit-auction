import pandas as pd
def print_leaderboard(df, current_step=None, total_steps=None):
    """
    Helper function to print the current market stats in a clean table.
    """
    if df.empty:
        print("No transactions recorded yet.")
        return

    # Header
    if current_step and total_steps:
        print(f"\n📊 INTERMEDIATE LEADERBOARD [Prompts: {current_step}/{total_steps}]")
    else:
        print("\n" + "="*80)
        print("🏆 FINAL MARKET LEADERBOARD")
        print("="*80)

    # Aggregation
    stats = df.groupby("winner").agg({
        "score": ["count", "mean"], 
        "profit": "sum", 
        "latency": "mean"
    })
    
    # Formatting
    stats.columns = ["Wins", "Avg Score", "Total Profit", "Avg Latency"]
    stats = stats.sort_values(by="Total Profit", ascending=False)
    
    # Print
    print("-" * 80)
    print(stats.to_string(float_format=lambda x: "{:.4f}".format(x)))
    print("-" * 80)
    
    if current_step:
        print(f"💰 Session Profit So Far: ${df['profit'].sum():.6f}")
        print("-" * 80 + "\n")
    else:
        print(f"💰 TOTAL SESSION PROFIT: ${df['profit'].sum():.6f}")