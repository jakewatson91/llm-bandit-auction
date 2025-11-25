import os 
from litellm import completion
import json
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

### Models

bidding_agents = [
    "groq/qwen/qwen3-32b",             
    "groq/llama-3.1-8b-instant",
    "groq/llama-3.3-70b-versatile",
    "groq/meta-llama/llama-4-maverick-17b-128e-instruct", # Beta
    "groq/meta-llama/llama-4-scout-17b-16e-instruct",     # Beta
    "groq/moonshotai/kimi-k2-instruct-0905",
    "groq/openai/gpt-oss-120b",
    "groq/openai/gpt-oss-20b"
]

# --- 1. The Agent (Bidder) ---
def get_agent_bid(model_id, user_prompt):
    """
    Asks the agent to estimate its confidence and place a bid.
    Returns: Dict with {'confidence': float, 'bid': float, 'reasoning': str}
    """
    
    # The "Game Theory" System Prompt
    system_instruction = (
        "You are an AI agent in a competitive market. "
        "You must bid to answer the user's query.\n"
        "1. Analyze the difficulty of the query.\n"
        "2. Estimate your probability of answering correctly (0.0 to 1.0).\n"
        "3. Output ONLY valid JSON: {\"confidence\": 0.95, \"reasoning\": \"...\"}"
    )

    try:
        response = completion(
            model=model_id, 
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ],
            api_key=groq_api_key,
            response_format={"type": "json_object"} # Forces valid JSON
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        # If model fails to bid (common with smaller models), return default
        print(f"Bidding Error on {model_id}: {e}")
        return {"confidence": 0.0, "reasoning": "Failed to bid"}

# --- 2. The Execution (Winner Only) ---
def get_agent_answer(model_id, user_prompt):
    """Call this ONLY for the winning agent"""
    response = completion(
        model=model_id,
        messages=[{"role": "user", "content": user_prompt}],
        api_key=groq_api_key
    )
    return response.choices[0].message.content

# --- 3. The Judge (Evaluator) ---
def judge_answer(user_prompt, agent_response):
    """
    The unbiased judge. 
    Note: 'gemini-2.5' is not standard yet; defaulted to 'gemini-1.5-flash' 
    which is the current fast/free standard.
    """
    try:
        response = completion(
            model="gemini/gemini-1.5-flash", 
            messages=[{
                "role": "user", 
                "content": (
                    f"Rate this answer from 0.0 to 1.0. Output ONLY the number.\n"
                    f"Prompt: {user_prompt}\nAnswer: {agent_response}"
                )
            }],
            api_key=gemini_api_key
        )
        return float(response.choices[0].message.content.strip())
    except:
        return 0.5 # Fallback score