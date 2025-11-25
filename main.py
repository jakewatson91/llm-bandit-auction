import os
import json
import time
import numpy as np
import pandas as pd
from litellm import completion
from dotenv import load_dotenv

# --- 0. SETUP ---
load_dotenv()  # Reads .env for GROQ_API_KEY / GEMINI_API_KEY

# --- 1. CONFIGURATION ---
REWARD_PER_QUERY = 0.05
QUALITY_THRESHOLD = 0.7
FALLBACK_MODEL_ID = "groq/llama-3.3-70b-versatile"

# Pricing & Specs (from Groq: https://groq.com/pricing)
MODEL_CATALOG = {
    "groq/openai/gpt-oss-20b": {
        "cost_in": 0.075 / 1e6, "cost_out": 0.30 / 1e6, "tps": 1000
    },
    "groq/openai/gpt-oss-120b": {
        "cost_in": 0.15 / 1e6, "cost_out": 0.60 / 1e6, "tps": 500
    },
    "groq/moonshotai/kimi-k2-instruct-0905": {
        "cost_in": 1.00 / 1e6, "cost_out": 3.00 / 1e6, "tps": 200
    },
    "groq/meta-llama/llama-4-scout-17b-16e-instruct": {
        "cost_in": 0.11 / 1e6, "cost_out": 0.34 / 1e6, "tps": 594
    },
    "groq/meta-llama/llama-4-maverick-17b-128e-instruct": {
        "cost_in": 0.20 / 1e6, "cost_out": 0.60 / 1e6, "tps": 562
    },
    "groq/qwen/qwen3-32b": {
        "cost_in": 0.29 / 1e6, "cost_out": 0.59 / 1e6, "tps": 662
    },
    "groq/llama-3.3-70b-versatile": {
        "cost_in": 0.59 / 1e6, "cost_out": 0.79 / 1e6, "tps": 394
    },
    "groq/llama-3.1-8b-instant": {
        "cost_in": 0.05 / 1e6, "cost_out": 0.08 / 1e6, "tps": 840
    }
}

# --- 2. AGENT CLASS (THE BRAIN) ---
class MarketAgent:
    def __init__(self, model_id):
        self.id = model_id
        self.bankroll = 10.00
        
        if model_id not in MODEL_CATALOG:
            raise ValueError(f"CRITICAL ERROR: Model '{model_id}' not found in catalog. Check your spelling.")
            
        self.specs = MODEL_CATALOG[model_id]
        
        # Pure Thompson Sampling Priors
        self.alpha = 1.0 
        self.beta = 1.0

    def calculate_bid(self, self_assessment_score):
        """
        Pure Thompson Sampling modulated by Context.
        
        self_assessment_score: float (0.0 to 1.0) - How confident the LLM feels about THIS prompt.
        """
        # 1. Sample from History (The "General Skill")
        # "How good have I been in the past?"
        historical_prob = np.random.beta(self.alpha, self.beta)
        
        # 2. Combine with Context (The "Specific Confidence")
        # Probability Chain rule: P(Win) = P(Good_Generally) * P(Good_At_This)
        final_probability = historical_prob * self_assessment_score
        
        # 3. Calculate Estimated Cost
        # Estimate: 500 input tokens + 200 output tokens
        est_cost = (500 * self.specs['cost_in']) + (200 * self.specs['cost_out'])
        
        # 4. Expected Value
        # EV = (Prob_Win * Reward) - Cost
        expected_value = (final_probability * REWARD_PER_QUERY) - est_cost
        
        # 5. The Bid (Floored at 0)
        bid = max(0.0, expected_value)
        
        return bid, est_cost

    def update(self, score, cost):
        """Standard Bayesian Update"""
        # 1. Update Beliefs
        self.alpha += score
        self.beta += (1.0 - score)
        
        # 2. Update Bankroll (Revenue is proportional to quality)
        revenue = REWARD_PER_QUERY * score
        profit = revenue - cost
        self.bankroll += profit
        
        return profit

# --- 3. NETWORK HELPERS ---

def get_self_assessment(model_id, prompt):
    """
    Asks the agent to rate the difficulty of the prompt.
    Returns a float 0.0 (Impossible) to 1.0 (Easy/Confident).
    """
    system_msg = (
        "You are a bidding agent. Analyze the difficulty of this query for your capabilities. "
        "Return valid JSON containing 'confidence_score' (0.0 to 1.0). "
        "0.0 = I cannot answer this. 1.0 = I can answer perfectly."
    )
    
    try:
        response = completion(
            model=model_id,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1 # Low temp for consistent analysis
        )
        data = json.loads(response.choices[0].message.content)
        return float(data.get("confidence_score", 0.5))
    except Exception as e:
        # If model is dumb/fails to output JSON, assume low confidence
        return 0.1

def get_agent_answer(model_id, prompt):
    """Generates the actual response"""
    try:
        response = completion(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return "Error generating response."

def judge_answer(prompt, answer):
    """The Unbiased Judge (Gemini)"""
    try:
        response = completion(
            model="gemini/gemini-1.5-flash",
            messages=[{
                "role": "user", 
                "content": (
                    f"Rate this answer from 0.0 to 1.0 based on accuracy. "
                    f"Output ONLY the number.\n\nPrompt: {prompt}\nAnswer: {answer}"
                )
            }]
        )
        return float(response.choices[0].message.content.strip())
    except:
        return 0.5

# --- 4. MAIN SIMULATION LOOP ---

def run_simulation(agents_list, prompts):
    # Initialize Agents
    market = [MarketAgent(mid) for mid in agents_list]
    logs = []

    print(f"\n--- MARKET OPEN: {len(market)} AGENTS ---")
    print(f"--- REWARD: ${REWARD_PER_QUERY} | THRESHOLD: {QUALITY_THRESHOLD} ---")

    for round_i, prompt in enumerate(prompts):
        print(f"\n=== Round {round_i+1}: '{prompt[:40]}...' ===")
        
        # --- A. BIDDING PHASE ---
        bids = []
        for agent in market:
            if agent.bankroll <= 0: continue # Bankrupt

            # 1. Get Context (Self-Assessment)
            # "Does this look hard?"
            self_score = get_self_assessment(agent.id, prompt)
            
            # 2. Calculate Bid (Bandit History * Context)
            bid_amt, est_cost = agent.calculate_bid(self_score)
            
            if bid_amt > 0:
                # Calculate mean just for display (alpha / alpha+beta)
                historical_mean = agent.alpha / (agent.alpha + agent.beta)
                print(f"  > {agent.id.split('/')[-1]}: "
                      f"Hist={historical_mean:.2f}, Self={self_score:.2f} -> Bid=${bid_amt:.5f}")
                
                bids.append({
                    "agent": agent, 
                    "bid": bid_amt, 
                    "cost": est_cost
                })

        if not bids:
            print("  [!] No bids. Market failed.")
            continue

        # --- B. AUCTION RESOLUTION (Vickrey) ---
        bids.sort(key=lambda x: x['bid'], reverse=True)
        winner = bids[0]
        # Second price rule
        second_price = bids[1]['bid'] if len(bids) > 1 else 0.00001
        
        print(f"  *** WINNER: {winner['agent'].id.split('/')[-1]} (Pays ${second_price:.5f})")

        # --- C. EXECUTION & JUDGMENT ---
        raw_answer = get_agent_answer(winner['agent'].id, prompt)
        score = judge_answer(prompt, raw_answer)
        
        # --- D. RETRY GATE (The Safety Net) ---
        actual_cost = second_price + winner['cost']
        final_profit = 0.0
        accepted = False

        if score < QUALITY_THRESHOLD:
            print(f"  [X] REJECTED (Score {score:.2f} < {QUALITY_THRESHOLD})")
            print(f"  [>] Redirecting to {FALLBACK_MODEL_ID.split('/')[-1]}...")
            
            # Punish the agent: Full Cost, Zero Revenue
            # We treat this as a "Total Loss" for the bandit (score=0.0)
            final_profit = winner['agent'].update(score=0.0, cost=actual_cost)
            
        else:
            print(f"  [✓] ACCEPTED (Score {score:.2f})")
            # Reward the agent: Cost + Revenue proportional to score
            final_profit = winner['agent'].update(score=score, cost=actual_cost)
            accepted = True

        print(f"  *** PROFIT: ${final_profit:.5f} | BANKROLL: ${winner['agent'].bankroll:.4f}")

        # --- E. LOGGING ---
        logs.append({
            "round": round_i,
            "prompt": prompt,
            "winner": winner['agent'].id,
            "bid": winner['bid'],
            "score": score,
            "accepted": accepted,
            "profit": final_profit,
            "bankroll": winner['agent'].bankroll
        })
        
        time.sleep(0.5) # Slight delay to avoid rate limits

    return pd.DataFrame(logs)

# --- 5. EXECUTION ---
if __name__ == "__main__":
    # Define Agents
    active_agents = list(MODEL_CATALOG.keys())
    
    # Define Prompts (Mix of Easy/Hard)
    test_prompts = [
        "What is 2 + 2?",
        "Explain Quantum Chromodynamics in detail.",
        "Write a python function to reverse a list.",
        "Who was the first US president?",
        "Write a 500 word essay on the fall of Rome.",
        "What color is the sky?",
        "Solve this differential equation: dy/dx = y * x",
        "Translate 'Hello' to Spanish.",
        "Analyze the nuances of 19th century French poetry.",
        "What is the capital of France?"
    ]

    df = run_simulation(active_agents, test_prompts)
    
    print("\n--- FINAL RESULTS ---")
    print(df[["winner", "bid", "score", "accepted", "profit"]])