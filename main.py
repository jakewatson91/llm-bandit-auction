import os
import json
import time
import argparse
import numpy as np
import pandas as pd
from litellm import completion
from dotenv import load_dotenv
from data_loader import get_benchmark_prompts

# --- 0. SETUP ---
load_dotenv()

# --- 1. CONFIGURATION ---
REWARD_PER_QUERY = 0.15
TIME_PENALTY_PER_SEC = 0.002
QUALITY_THRESHOLD = 0.7
WEIGHTS_FILE = "market_weights.json"

# JUDGE: Gemini 2.5 Flash (Strictly)
JUDGE_MODEL_ID = "gemini/gemini-2.5-flash-lite"

VALID_CATEGORIES = ["general", "medical", "law", "coding", "math", "finance"]

# Specs
MODEL_CATALOG = {
    "groq/llama-3.1-8b-instant": {
        "cost_in": 0.05 / 1e6, "cost_out": 0.08 / 1e6, "tps": 840, "base_latency": 0.2
    },
    "groq/openai/gpt-oss-20b": {
        "cost_in": 0.075 / 1e6, "cost_out": 0.30 / 1e6, "tps": 1000, "base_latency": 0.2
    },
    "groq/meta-llama/llama-4-scout-17b-16e-instruct": {
        "cost_in": 0.11 / 1e6, "cost_out": 0.34 / 1e6, "tps": 594, "base_latency": 0.25
    },
    "groq/qwen/qwen3-32b": {
        "cost_in": 0.29 / 1e6, "cost_out": 0.59 / 1e6, "tps": 662, "base_latency": 0.25
    },
    "groq/llama-3.3-70b-versatile": {
        "cost_in": 0.59 / 1e6, "cost_out": 0.79 / 1e6, "tps": 300, "base_latency": 0.5
    },
    "groq/openai/gpt-oss-120b": {
        "cost_in": 0.15 / 1e6, "cost_out": 0.60 / 1e6, "tps": 500, "base_latency": 0.3
    },
    "groq/moonshotai/kimi-k2-instruct-0905": {
        "cost_in": 1.00 / 1e6, "cost_out": 3.00 / 1e6, "tps": 200, "base_latency": 0.6
    }
}

# --- 2. AGENT LOGIC ---
class MarketAgent:
    def __init__(self, model_id):
        self.id = model_id
        self.bankroll = 10.00
        self.specs = MODEL_CATALOG[model_id]
        
        # Session Stats
        self.session_spend = 0.0
        self.session_revenue = 0.0
        self.wins = 0
        self.losses = 0
        
        # Learning Weights
        self.skills = {cat: {'alpha': 1.0, 'beta': 1.0} for cat in VALID_CATEGORIES}

    def export_weights(self):
        return {
            "bankroll": self.bankroll,
            "skills": self.skills,
            "stats": {
                "wins": self.wins,
                "losses": self.losses,
                "spend": self.session_spend,
                "revenue": self.session_revenue
            }
        }

    def load_weights(self, data):
        self.bankroll = data.get("bankroll", 10.00)
        saved_skills = data.get("skills", {})
        for cat, weights in saved_skills.items():
            if cat in self.skills:
                self.skills[cat] = weights
        
        # Load historical stats
        stats = data.get("stats", {})
        self.wins = stats.get("wins", 0)
        self.losses = stats.get("losses", 0)
        self.session_spend = stats.get("spend", 0.0)
        self.session_revenue = stats.get("revenue", 0.0)

    def calculate_bid(self, self_confidence, category, input_char_len, est_output_tokens):
        cat = category if category in self.skills else "general"
        alpha = self.skills[cat]['alpha']
        beta = self.skills[cat]['beta']
        
        # 1. Bandit
        historical_prob = np.random.beta(alpha, beta)
        final_prob = historical_prob * self_confidence
        
        # 2. Economics
        est_tokens = 150
        est_latency = self.specs['base_latency'] + (est_tokens / self.specs['tps'])
        est_input_tokens = input_char_len / 4.0
        
        compute_cost = (est_input_tokens * self.specs['cost_in']) + (est_tokens * self.specs['cost_out'])
        time_penalty = est_latency * TIME_PENALTY_PER_SEC
        total_cost = compute_cost + time_penalty
        
        # 3. Bid
        expected_revenue = final_prob * REWARD_PER_QUERY
        bid = max(0.0, expected_revenue - total_cost)
        return bid, compute_cost

    def update(self, score, total_real_cost, actual_latency, category):
        cat = category if category in self.skills else "general"
        
        # Learning
        self.skills[cat]['alpha'] += score
        self.skills[cat]['beta'] += (1.0 - score)
        
        # Accounting
        gross_revenue = REWARD_PER_QUERY * score
        time_penalty = actual_latency * TIME_PENALTY_PER_SEC
        net_revenue = max(0.0, gross_revenue - time_penalty)
        
        profit = net_revenue - total_real_cost
        
        self.bankroll += profit
        self.session_spend += total_real_cost
        self.session_revenue += net_revenue
        
        if profit > 0: self.wins += 1
        else: self.losses += 1
            
        # RETURN TUPLE FOR LOGGING
        return profit, gross_revenue, time_penalty

# --- 3. SYSTEM FUNCTIONS ---
def initialize_market():
    agents = []
    loaded_data = {}
    if os.path.exists(WEIGHTS_FILE):
        try:
            with open(WEIGHTS_FILE, 'r') as f:
                loaded_data = json.load(f)
            print(f"[DISK] Loaded weights from {WEIGHTS_FILE}")
        except: pass

    for model_id in MODEL_CATALOG.keys():
        agent = MarketAgent(model_id)
        if model_id in loaded_data:
            agent.load_weights(loaded_data[model_id])
        agents.append(agent)
    return agents

def save_market(agents):
    data = {agent.id: agent.export_weights() for agent in agents}
    try:
        with open(WEIGHTS_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\n[DISK] Saved updated weights to {WEIGHTS_FILE}")
    except Exception as e:
        print(f"[DISK] Failed to save weights: {e}")

# --- 4. HELPERS ---
def classify_and_assess(model_id, prompt):
    system_msg = (
        f"Classify into one: {json.dumps(VALID_CATEGORIES)}.\n"
        "Rate confidence (0.0 to 1.0).\n"
        "Estimate tokens (e.g. 50).\n"
        "Return JSON: {\"category\": \"...\", \"confidence\": 0.9, \"tokens\": 50}"
    )
    try:
        response = completion(
            model=model_id,
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        data = json.loads(response.choices[0].message.content)
        cat = data.get("category", "general").lower()
        if cat not in VALID_CATEGORIES: cat = "general"
        return float(data.get("confidence", 0.5)), cat, int(data.get("tokens", 100))
    except:
        return 0.1, "general", 100

def get_agent_answer(model_id, prompt):
    try:
        response = completion(
            model=model_id, messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except: return "Error"

def judge_answer(prompt, answer):
    try:
        response = completion(
            model=JUDGE_MODEL_ID,
            messages=[{
                "role": "user", 
                "content": f"Rate accuracy 0.0 to 1.0. Be critical and objective. Return JSON: {{\"score\": 0.9}}\n\nPrompt: {prompt}\nAnswer: {answer}"
            }],
            response_format={"type": "json_object"}
        )
        data = json.loads(response.choices[0].message.content)
        return float(data.get("score", 0.0))
    except Exception as e:
        print(f"  [!] Judge Error: {e}")
        return 0.0

# --- 5. MAIN SIMULATION ---
def run_simulation(market_agents, prompts):
    logs = []
    print(f"\n--- MARKET OPEN ({len(market_agents)} Agents) ---")

    for round_i, prompt_data in enumerate(prompts):
        prompt_text = prompt_data['text']
        true_cat = prompt_data['cat']
        input_len = len(prompt_text)
        
        print(f"\n=== ROUND {round_i+1} ===")
        print(f"Prompt: '{prompt_text[:50]}...' ({true_cat})")
        
        bids = []
        for agent in market_agents:
            if agent.bankroll <= 0: continue

            conf, cat, est_toks = classify_and_assess(agent.id, prompt_text)
            bid_amt, cost = agent.calculate_bid(conf, cat, input_len, est_toks)
            
            if bid_amt > 0:
                bids.append({"agent": agent, "bid": bid_amt, "cost": cost, "cat": cat})
                print(f"  > {agent.id.split('/')[-1][:15]:<15} | Bid:${bid_amt:.4f}")

        if not bids:
            print("  [!] Market Failure: No bids.")
            continue

        bids.sort(key=lambda x: x['bid'], reverse=True)
        winner = bids[0]
        second_price = bids[1]['bid'] if len(bids) > 1 else 0.00001
        
        print(f"  --> WINNER: {winner['agent'].id.split('/')[-1]} (Pays ${second_price:.4f})")

        start = time.time()
        raw_answer = get_agent_answer(winner['agent'].id, prompt_text)
        duration = time.time() - start
        
        print("  ... Judging ...")
        score = judge_answer(prompt_text, raw_answer)
        
        # Real Accounting
        actual_out_tokens = len(raw_answer) / 4.0
        actual_in_tokens = input_len / 4.0
        real_compute_cost = (actual_in_tokens * winner['agent'].specs['cost_in']) + \
                            (actual_out_tokens * winner['agent'].specs['cost_out'])
        total_real_bill = second_price + real_compute_cost

        # Update and Unpack Financials
        if score < QUALITY_THRESHOLD:
            print(f"  [X] REJECTED (Score {score:.2f})")
            profit, gross, penalty = winner['agent'].update(0.0, total_real_bill, duration, winner['cat'])
        else:
            print(f"  [✓] ACCEPTED (Score {score:.2f})")
            profit, gross, penalty = winner['agent'].update(score, total_real_bill, duration, winner['cat'])

        # --- DETAILED LOGGING ---
        print(f"      Latency: {duration:.4f}s  (Penalty: -${penalty:.4f})")
        print(f"      Revenue: +${gross:.4f}")
        print(f"      Bill   : -${total_real_bill:.4f}")
        print(f"      PROFIT : ${profit:.4f}")

        logs.append({
            "winner": winner['agent'].id, 
            "category": winner['cat'], 
            "score": score, 
            "profit": profit
        })
        time.sleep(0.5)

    return pd.DataFrame(logs)

# --- 6. EXECUTION ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run without saving weights")
    args = parser.parse_args()

    # 1. Initialize
    market = initialize_market()
    
    # 2. Load Data
    try:
        test_prompts = get_benchmark_prompts(n_total=100)
    except ImportError:
        print("Warning: data_loader.py not found. Using manual fallback.")
        test_prompts = []
    except Exception as e:
        print(f"Warning: Loader failed: {e}")
        test_prompts = []

    # Fallback if loader returns nothing
    if not test_prompts:
        print("Using manual fallback list.")
        test_prompts = [
            {"text": "What is 2 + 2?", "cat": "math"},
            {"text": "Write a poem about rust.", "cat": "general"},
            {"text": "Solve 5x = 20", "cat": "math"},
            {"text": "Explain quantum physics.", "cat": "general"}
        ]

    # 3. Run
    df = run_simulation(market, test_prompts)
    
    # 4. Save
    if not args.test:
        save_market(market)
    else:
        print("\n[TEST MODE] Weights NOT saved.")
    
    # 5. Report
    print("\n" + "="*60)
    print("FINAL FINANCIAL REPORT")
    print("="*60)
    
    leaderboard = []
    for agent in market:
        short_name = agent.id.split('/')[-1]
        net_profit = agent.session_revenue - agent.session_spend
        roi = (net_profit / agent.session_spend * 100) if agent.session_spend > 0 else 0.0
        
        leaderboard.append({
            "Agent": short_name,
            "Wins": agent.wins,
            "Losses": agent.losses,
            "Spend": f"${agent.session_spend:.4f}",
            "Profit": f"${net_profit:.4f}",
            "ROI": f"{roi:.1f}%",
            "Bankroll": f"${agent.bankroll:.2f}"
        })
        
    lb_df = pd.DataFrame(leaderboard)
    print(lb_df.to_string(index=False))