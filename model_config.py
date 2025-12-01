# model_config.py

VALID_CATEGORIES = ["general", "medical", "law", "coding", "math", "finance"]

# ---------------- TRAIN / MARKET MODELS ----------------
MODEL_CATALOG = {
    "llama-3.1-8b-instant": {
        "provider": "groq",
        "cost_in": 0.05 / 1e6, "cost_out": 0.08 / 1e6, "tps": 840, "base_latency": 0.2
    },
    "openai/gpt-oss-20b": {
        "provider": "groq",
        "cost_in": 0.075 / 1e6, "cost_out": 0.30 / 1e6, "tps": 1000, "base_latency": 0.2
    },
    "meta-llama/llama-4-scout-17b-16e-instruct": {
        "provider": "groq", 
        "cost_in": 0.11 / 1e6, "cost_out": 0.34 / 1e6, "tps": 594, "base_latency": 0.25
    },
    "qwen/qwen3-32b": {
        "provider": "groq",
        "cost_in": 0.29 / 1e6, "cost_out": 0.59 / 1e6, "tps": 662, "base_latency": 0.25
    },
    "llama-3.3-70b-versatile": {
        "provider": "groq",
        "cost_in": 0.59 / 1e6, "cost_out": 0.79 / 1e6, "tps": 300, "base_latency": 0.5
    },
    "openai/gpt-oss-120b": {
        "provider": "groq",
        "cost_in": 0.15 / 1e6, "cost_out": 0.60 / 1e6, "tps": 500, "base_latency": 0.3
    },
    "moonshotai/kimi-k2-instruct-0905": {
        "provider": "groq",
        "cost_in": 1.00 / 1e6, "cost_out": 3.00 / 1e6, "tps": 200, "base_latency": 0.6
    }
}

# ---------------- BENCHMARK MODELS ----------------
BENCHMARK_MODELS = {
    "llama-3.3-70b-versatile": {
        "provider": "groq",
        "cost_in": 0.59 / 1e6, "cost_out": 0.79 / 1e6
    },
    "gemini-2.5-pro": { 
        "provider": "gemini",
        "cost_in": 1.25 / 1e6, 
        "cost_out": 10.00 / 1e6,
        "throttle_delay": 30 
    },
    "gpt-4o": {
        "provider": "openai",
        "cost_in": 2.50 / 1e6, "cost_out": 10.00 / 1e6
    }
}

# ---------------- FALLBACK ----------------
FALLBACK_CONFIG = {
    "provider": "groq",
    "model": "llama-3.3-70b-versatile"
}

# ---------------- JUDGES ----------------
JUDGE_MODELS = [
    {"provider": "gemini", "model": "gemini-2.5-flash"},
    {"provider": "gemini", "model": "gemini-2.5-flash-lite"},
    {"provider": "gemini", "model": "gemini-2.0-flash"}
]

# ---------------- RETRY & SLEEP SETTINGS ----------------
RETRY_CONFIG = {
    "max_retries": 3,
    "sleep_map": {
        "gemini": 30,
        "default": 2
    },
    "server_error_sleep": 5,
    "general_error_sleep": 2
}