import sys
import os
import json
import time
import argparse
import re
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from data_loader import get_benchmark_prompts

from model_config import (
    MODEL_CATALOG, VALID_CATEGORIES, 
    FALLBACK_CONFIG, JUDGE_MODELS
)
from prompts import JUDGE_PROMPT
from models import call_model

load_dotenv()

REWARD_PER_QUERY = 0.15
TIME_PENALTY_PER_SEC = 0.004
QUALITY_THRESHOLD = 0.8
WEIGHTS_FILE = "market_weights.json"
CURRENT_JUDGE_INDEX = 0

# ---------------- HELPER FUNCTIONS ----------------

def classify_and_assess(model_id, provider, prompt):
    system_msg = (
        f"Classify into one: {json.dumps(VALID_CATEGORIES)}.\n"
        "Rate confidence (0.0 to 1.0).\n"
        "Estimate tokens.\n"
        "Return JSON: {\"category\": \"...\", \"confidence\": 0.9, \"tokens\": 100}"
    )
    try:
        raw = call_model(provider, model_id, prompt, system=system_msg)
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            data = json.loads(raw)
        
        cat = data.get("category", "general")
        if cat not in VALID_CATEGORIES: cat = "general"
        return float(data.get("confidence", 0.5)), cat, int(data.get("tokens", 100))
    except Exception:
        return 0.5, "general", 100

def judge_answer(prompt, answer):
    global CURRENT_JUDGE_INDEX
    
    for i in range(len(JUDGE_MODELS)):
        idx = (CURRENT_JUDGE_INDEX + i) % len(JUDGE_MODELS)
        judge_conf = JUDGE_MODELS[idx]
        
        # Double Tap Logic
        for attempt in range(2):
            try:
                msg = f"{prompt}\n\nAnswer: {answer}\n\n{JUDGE_PROMPT}"
                
                raw = call_model(judge_conf['provider'], judge_conf['model'], msg)
                match = re.search(r"\{.*\}", raw, re.DOTALL)
                if match:
                    score = float(json.loads(match.group()).get("score", 0.0))
                    CURRENT_JUDGE_INDEX = idx
                    return score
            except Exception:
                if attempt == 0:
                    time.sleep(1)
                    continue
                pass 
                
    return 0.0

# ---------------- AGENT ----------------

class MarketAgent:
    def __init__(self, model_id):
        self.id = model_id
        self.specs = MODEL_CATALOG[model_id]
        self.provider = self.specs['provider']
        self.bankroll = 10.0
        self.wins = 0
        self.losses = 0
        self.session_revenue = 0.0
        self.session_spend = 0.0
        self.skills = {c: {"alpha": 1.0, "beta": 1.0} for c in VALID_CATEGORIES}

    def export_weights(self):
        return {"bankroll": self.bankroll, "skills": self.skills, "stats": {"wins": self.wins}}

    def load_weights(self, d):
        self.bankroll = d.get("bankroll", 10.0)
        self.skills.update(d.get("skills", {}))
        self.wins = d.get("stats", {}).get("wins", 0)

    def calculate_bid(self, conf, cat, input_len, est_tokens):
        stats = self.skills.get(cat, self.skills["general"])
        # Standard Probability calc
        prob = np.random.beta(stats["alpha"], stats["beta"]) * conf
        
        cost_compute = (input_len/4 * self.specs["cost_in"]) + (est_tokens * self.specs["cost_out"])
        
        lat_est = self.specs["base_latency"] + (est_tokens / self.specs["tps"])
        cost_time = lat_est * TIME_PENALTY_PER_SEC
        
        bid = max(0.0, (prob * REWARD_PER_QUERY) - (cost_compute + cost_time))
        return bid, cost_compute

    def update(self, score, real_cost, latency, cat):
        if cat in self.skills:
            self.skills[cat]["alpha"] += score
            self.skills[cat]["beta"] += (1.0 - score)
        gross = score * REWARD_PER_QUERY
        penalty = latency * TIME_PENALTY_PER_SEC
        net = max(0.0, gross - penalty)
        profit = net - real_cost
        
        self.bankroll += profit
        self.session_revenue += net
        self.session_spend += real_cost
        
        if profit > 0: self.wins += 1
        else: self.losses += 1

# ---------------- SIMULATION ----------------

def initialize_market(load_weights_file):
    saved = {}
    if load_weights_file and os.path.exists(WEIGHTS_FILE):
        with open(WEIGHTS_FILE) as f: 
            saved = json.load(f)
        print(f"Loaded weights for {len(saved)} agents.")
    agents = []
    for mid in MODEL_CATALOG:
        a = MarketAgent(mid)
        if mid in saved: a.load_weights(saved[mid])
        agents.append(a)
    return agents

def run_simulation(agents, prompts, train=True, evaluate=False, verbose=False):
    logs = []
    print(f"\n--- SIMULATION START ({len(prompts)} Prompts) ---")

    for i, p in enumerate(prompts):
        text = p['text']
        
        if not evaluate:
            if verbose:
                print(f"PROMPT [{i+1}]: {text}...")
            else:
                print(f"PROMPT [{i+1}]: {text[:80]}...")
            print(f"{'-'*60}")
        
        # 1. Bids
        bids_data = [] # (bid_val, agent, category)
        current_wallets = {a.id: a.bankroll for a in agents} 
        
        for a in agents:
            if a.bankroll < 0: continue
            conf, cat, toks = classify_and_assess(a.id, a.provider, text)
            bid, _ = a.calculate_bid(conf, cat, len(text), toks)
            if bid > 0: bids_data.append((bid, a, cat))
        
        if not bids_data:
            if train: print("  > No bids placed.")
            continue
            
        bids_data.sort(key=lambda x: x[0], reverse=True)
        
        bid_val, winner, cat = bids_data[0]
        second_price = bids_data[1][0] if len(bids_data) > 1 else 0.001
        
        if not evaluate:
            print("BIDDING WAR:")
            for b_val, b_agent, b_cat in bids_data:
                wallet_bal = current_wallets.get(b_agent.id, b_agent.bankroll)
                print(f"  • {b_agent.id:<35} | Bid: ${b_val:.6f} | Wallet: ${wallet_bal:.6f}")
            print(f"{'-'*60}")
            print(f"🏆 WINNER: {winner.id} (Bid: ${bid_val:.6f} | 2nd: ${second_price:.6f})")
        
        # 2. Exec
        start = time.time()
        try:
            answer = call_model(winner.provider, winner.id, text)
        except Exception as e:
            if train: print(f"[ERROR] Execution failed: {e}")
            answer = "Error"
        if verbose:
            print(f"  > Answer: {answer}")
        lat = time.time() - start
        
        # 3. Judge
        score = judge_answer(text, answer)
        
        # 4. Settlement
        real_cost_compute = (len(text)/4 * winner.specs["cost_in"]) + (len(answer)/4 * winner.specs["cost_out"])
        bill = second_price + real_cost_compute
        lat_penalty = lat * TIME_PENALTY_PER_SEC
        
        is_failure = score < QUALITY_THRESHOLD
        effective_score = 0.0 if is_failure else score
        revenue = effective_score * REWARD_PER_QUERY
        final_profit = revenue - (lat_penalty + bill)

        if not evaluate:
            print(f"📊 REPORT:")
            print(f"  • Latency:       {lat:.3f}s (Penalty: -${lat_penalty:.6f})")
            print(f"  • Score:         {score:.2f} " + ("[FAIL] -> Effective Revenue: $0.00" if is_failure else f"[PASS] -> Revenue: +${revenue:.6f}"))
            print(f"  ------------------------------------------------")
            print(f"  + Revenue:       ${revenue:.6f}")
            print(f"  - Latency Cost:  ${lat_penalty:.6f}")
            print(f"  - 2nd Price:     ${second_price:.6f}")
            print(f"  - Compute Cost:  ${real_cost_compute:.6f}")
            print(f"  ------------------------------------------------")
            print(f"  = NET PROFIT:    ${final_profit:.6f}")
        
        if is_failure:
            if train: print(f"  ⚠️  [FAIL] Triggering fallback...")
            try:
                call_model(FALLBACK_CONFIG['provider'], FALLBACK_CONFIG['model'], text)
            except: pass
            winner.update(0.0, bill, lat, cat)
        else:
            winner.update(score, bill, lat, cat)
            
        logs.append({
            "winner": winner.id, 
            "category": cat, 
            "score": score, 
            "profit": final_profit, 
            "latency": lat,
            "compute_cost": real_cost_compute 
        })
        
        if not evaluate:
            print(f"\n💵 SETTLEMENT: {winner.id} New Bankroll: ${winner.bankroll:.6f}")
            print(f"{'='*60}\n")


    return pd.DataFrame(logs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--n_train", type=int, default=5)
    parser.add_argument("--verbose", action="store_true", help="prints full prompts and answers")
    args = parser.parse_args()
    
    prompts = get_benchmark_prompts(n_total=args.n_train)

    market = initialize_market(load_weights_file=(not args.test))
    df = run_simulation(market, prompts, train=(not args.test), verbose=args.verbose)
    
    if not args.test:
        with open(WEIGHTS_FILE, "w") as f:
            json.dump({a.id: a.export_weights() for a in market}, f, indent=2)
            
    print("\n" + "="*80)
    print("🏆 MARKET LEADERBOARD")
    print("="*80)
    if not df.empty:
        stats = df.groupby("winner").agg({"score": ["count", "mean"], "profit": "sum"})
        stats.columns = ["Wins", "Avg Score", "Total Profit"]
        stats = stats.sort_values(by="Total Profit", ascending=False)
        print(stats.to_string())
        print("-" * 80)
        print("\n📝 TRANSACTION LOG:")
        print(df.to_string(index=False))
        print("-" * 80)
        print(f"💰 TOTAL SESSION PROFIT: ${df['profit'].sum():.6f}")
    else:
        print("No transactions recorded.")
    print("="*80 + "\n")