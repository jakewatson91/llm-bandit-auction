import sys
import os
import json
import math
import time
import argparse
import re
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from data_loader import get_benchmark_prompts
from model_config import MODEL_CATALOG
from prompts import CLASSIFIER_PROMPT, JUDGE_PROMPT
from models import call_model_with_retry
from utils import print_leaderboard

load_dotenv()

# --- CONSTANTS ---
TIME_PENALTY_PER_SEC = 0.002 
QUALITY_THRESHOLD = 0.7
WEIGHTS_FILE = "market_weights_dec9_new_rewards.json"
CURRENT_JUDGE_INDEX = 0
CLASSIFIER_INDEX = 0
MAX_CONSECUTIVE_FAILURES = 3
EMA_ALPHA = 0.2
PRINT_SAVE_INTERVAL = 20

CATEGORIES = {
    "easy": { "output_est": 50, "reward": 0.02 },
    "hard": { "output_est": 500, "reward": 0.15 }
}

# ---------------- HELPER FUNCTIONS ----------------

def classify_prompt(prompt):
    global CLASSIFIER_INDEX
    classifiers = [k for k, v in MODEL_CATALOG.items() if v.get("is_classifier")]
    if not classifiers: return "hard", 0.0, 0.0
    
    full_prompt = f"{CLASSIFIER_PROMPT}\n\nText: {prompt[:1000]}"
    
    for i in range(len(classifiers)):
        idx = (CLASSIFIER_INDEX + i) % len(classifiers)
        mid = classifiers[idx]
        config = MODEL_CATALOG[mid]
    
        res, latency = call_model_with_retry(config['provider'], mid, full_prompt)
        if res:
            try:        
                match = re.search(r"\{.*\}", res, re.DOTALL)
                if match:
                    data = json.loads(match.group())
                    comp = data.get("complexity", "hard")
                    if comp not in ["easy", "hard"]: comp = "hard"
                    cost = (len(full_prompt)/4 * config.get('cost_in', 0)) + \
                           (len(res)/4 * config.get('cost_out', 0))
                    return comp, cost, latency
            except Exception: pass
        next_classifier = classifiers[(idx + 1) % len(classifiers)]
        print(f"\nClassifier {mid} failed. Rotating to {next_classifier}...")
    return "hard", 0.0, 0.0

def judge_answer(prompt, answer):
    global CURRENT_JUDGE_INDEX
    judges = [k for k, v in MODEL_CATALOG.items() if v.get("is_judge")]
    if not judges: return None, 0.0

    for _ in range(len(judges)):
        mid = judges[CURRENT_JUDGE_INDEX]
        config = MODEL_CATALOG[mid]
        
        msg = f"User Input{prompt}\n\nAgent Answer: {answer}\n\nInstructions: {JUDGE_PROMPT}"
        cost = (len(msg)/4 * config.get('cost_in', 0)) + (100 * config.get('cost_out', 0))

        res, _ = call_model_with_retry(config['provider'], mid, msg, is_judge=True)
        if res:
            try:
                match = re.search(r"\{.*\}", res, re.DOTALL)
                if match:
                    return float(json.loads(match.group()).get("score", 0)), cost
            except Exception: pass

        last_judge = judges[CURRENT_JUDGE_INDEX]
        CURRENT_JUDGE_INDEX = (CURRENT_JUDGE_INDEX + 1) % len(judges)
        print(f"  ⚠️ Judge {last_judge} failed. Rotating to {judges[CURRENT_JUDGE_INDEX]}...")
    
    return None, 0.0

# ---------------- EV ROUTER AGENT ----------------

class ModelEV:
    def __init__(self, model_id):
        self.id = model_id
        self.specs = MODEL_CATALOG[model_id]
        self.provider = self.specs['provider']
        self.ema_latency = self.specs.get("base_latency", 2.0)
        
        self.skills = {
            "easy": {"alpha": 1.0, "beta": 1.0, "n": 0},
            "hard": {"alpha": 1.0, "beta": 1.0, "n": 0}
        }
        self.ema_cost_usd = {}

    def export_weights(self):
        return {"skills": self.skills, "ema_latency": self.ema_latency, "ema_cost_usd": self.ema_cost_usd}

    def load_weights(self, d):
        self.skills.update(d.get("skills", {}))
        self.ema_latency = d.get("ema_latency", self.specs.get("base_latency", 2.0))
        self.ema_cost_usd.update(d.get("ema_cost_usd", {}))

    def calculate_ev(self, complexity, input_len):
        stats = self.skills.get(complexity, self.skills["hard"])
        category_data = CATEGORIES.get(complexity, CATEGORIES["hard"])

        base_quality = np.random.beta(stats["alpha"], stats["beta"])
        
        predicted_quality = base_quality
        
        expected_revenue = predicted_quality * category_data["reward"]

        if complexity in self.ema_cost_usd:
            est_cost = self.ema_cost_usd[complexity]
        else:
            est_output = max(category_data["output_est"], input_len // 2) 
            est_cost = (input_len/4 * self.specs["cost_in"]) + (est_output * self.specs["cost_out"])

        est_time_cost = self.ema_latency * TIME_PENALTY_PER_SEC
        ev = expected_revenue - (est_cost + est_time_cost)
        
        return ev, predicted_quality, est_cost

    def update(self, score, latency, realized_cost, complexity):
        if latency > 0:
            self.ema_latency = (EMA_ALPHA * latency) + ((1 - EMA_ALPHA) * self.ema_latency)
        
        if complexity not in self.skills: complexity = "easy"
        
        if complexity in self.ema_cost_usd:
            self.ema_cost_usd[complexity] = (EMA_ALPHA * realized_cost) + ((1 - EMA_ALPHA) * self.ema_cost_usd[complexity])
        else:
            self.ema_cost_usd[complexity] = realized_cost
            
        stats = self.skills[complexity]
        
        if "n" not in stats: stats["n"] = 0
        stats["n"] += 1
        
        stats["alpha"] += score
        stats["beta"] += (1.0 - score)

# ---------------- SIMULATION ----------------

def initialize_market(load_weights_file):
    saved = {}
    if load_weights_file and os.path.exists(WEIGHTS_FILE):
        with open(WEIGHTS_FILE) as f: 
            saved = json.load(f)
        print(f"Loaded weights for {len(saved)} models.")
    
    models = []
    for mid, specs in MODEL_CATALOG.items():
        if not specs.get('market_model'): continue
        m = ModelEV(mid)
        if mid in saved: m.load_weights(saved[mid])
        models.append(m)
    return models

def save_and_exit(models, msg=""):
    print(f"\n🛑 {msg} Saving weights and exiting...")
    with open(WEIGHTS_FILE, "w") as f:
        json.dump({m.id: m.export_weights() for m in models}, f, indent=2)
    sys.exit(0)

def save_weights(models):
    print(f"💾 Saving weights for {len(models)} models to {WEIGHTS_FILE}...")
    with open(WEIGHTS_FILE, "w") as f:
        json.dump({m.id: m.export_weights() for m in models}, f, indent=2)
    print("✅ Save complete.")

def run_simulation(models, prompts, train=True, evaluate=False, verbose=False):
    logs = []
    consecutive_failures = 0
    
    print(f"\n--- BINARY EV ROUTING ({len(prompts)} Prompts) ---")

    for i, prompt_data in enumerate(prompts):
        prompt_text = prompt_data['text']
        
        if not evaluate:
            print(f"PROMPT [{i+1}]: {prompt_text[:80]}..." if not verbose else f"PROMPT [{i+1}]: {prompt_text}")
            print(f"{'-'*60}")
        
        complexity, classifier_cost, classifier_latency = classify_prompt(prompt_text)
        if not evaluate: print(f"Routing: {complexity.upper()}")

        candidates = []
        for m in models:
            ev, pred_quality, est_cost = m.calculate_ev(complexity, len(prompt_text))
            candidates.append((ev, m, pred_quality, est_cost))
        
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        if not evaluate:
            print("EV ANALYSIS:")
            for ev, m, qual, cost in candidates[:5]:
                print(f"  • {m.id:<35} | Qual: {qual:.2f} | Est.Cost: ${cost:.5f} | EV: {ev:.5f}")
            print(f"{'-'*60}")

        selected_model = None
        model_response = None
        
        for ev, m, _, _ in candidates:
            model_response, latency = call_model_with_retry(m.provider, m.id, prompt_text)
            if model_response is None:
                print(f"  ⚠️ {m.id} failed. Penalizing...")
                m.update(0.0, 0.0, 0.0, complexity) 
                continue 
            selected_model = m
            if not evaluate: print(f"🚀 SELECTED: {selected_model.id} (Latency: {latency:.3f}s)")
            break 

        if selected_model is None:
            print(f"  🛑 FATAL: All models failed.")
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES: save_and_exit(models, "All APIs failed")
            continue 

        quality_score, total_judge_cost = judge_answer(prompt_text, model_response)
        
        if quality_score is None:
            consecutive_failures += 1
            print(f"  ⚠️ FAIL: Judge crashed.")
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES: save_and_exit(models, "Judges broken")
            continue 
        
        token_cost = (len(prompt_text)/4 * selected_model.specs["cost_in"]) + \
                     (len(model_response)/4 * selected_model.specs["cost_out"])
        realized_compute_cost = token_cost + classifier_cost + total_judge_cost
        
        is_failure = quality_score < QUALITY_THRESHOLD
        reward = CATEGORIES.get(complexity, CATEGORIES["hard"])["reward"]
        revenue = (quality_score * reward) if not is_failure else 0.0
        net_profit = revenue - (realized_compute_cost + (latency * TIME_PENALTY_PER_SEC))

        if not evaluate:
            print(f"📊 REPORT:")
            print(f"  • Score:         {quality_score:.2f} " + ("[FAIL]" if is_failure else "[PASS]"))
            print(f"  • Real Cost:     ${realized_compute_cost:.6f}")
            print(f"  • Net Utility:   ${net_profit:.6f}")
            print(f"{'='*60}\n")
        else:
             print(f"   [{i+1}/{len(prompts)}] Selected: {selected_model.id:<25} | Score: {quality_score:.4f} | Latency: {latency:.4f}s", flush=True)

        selected_model.update(quality_score, latency, realized_compute_cost, complexity)
        consecutive_failures = 0
        
        logs.append({
            "winner": selected_model.id, 
            "category": "simple",
            "complexity": complexity,
            "score": quality_score, 
            "profit": net_profit, 
            "latency": latency + classifier_latency, 
            "compute_cost": realized_compute_cost
        })
        
        if (i + 1) % PRINT_SAVE_INTERVAL == 0:
            if train:
                save_weights(models)
                print(f"--- PERIODIC SAVE at Round {i+1} ---")
            print_leaderboard(pd.DataFrame(logs), i + 1, len(prompts))

        time.sleep(4) 

    return pd.DataFrame(logs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--n_train", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--start_index", type=int, default=0)
    args = parser.parse_args()
    
    prompts = get_benchmark_prompts(n_total=args.n_train, start_index=args.start_index)
    models = initialize_market(load_weights_file=(not args.test))
    
    df = pd.DataFrame()
    df = run_simulation(models, prompts, train=(not args.test), verbose=args.verbose)
        
    if not args.test:
        save_weights(models)

    if not df.empty:   
        print_leaderboard(df)
    print("="*80 + "\n")