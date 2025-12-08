import os

# model_config.py

MODEL_CATALOG = {
    # Judges
    "tngtech/deepseek-r1t2-chimera:free": {
        "provider": "openrouter",
        "market_model": False, "benchmark_model": False, "is_judge": True,
        "cost_in": 0.30 / 1e6, "cost_out": 1.20 / 1e6, "base_latency": 0.35 
    },
    "tngtech/deepseek-r1t-chimera:free": {
        "provider": "openrouter",
        "market_model": False, "benchmark_model": False, "is_judge": True,
        "cost_in": 0.30 / 1e6, "cost_out": 1.20 / 1e6, "base_latency": 0.35
    },
    "gemini-2.5-flash": {
        "provider": "gemini", 
        "market_model": False, "benchmark_model": False, "is_judge": True,
        "cost_in": 0.30 / 1e6, "cost_out": 2.50 / 1e6
    },
    "gemini-2.5-flash-lite": {
        "provider": "gemini", 
        "market_model": False, "benchmark_model": False, "is_judge": True,
        "cost_in": 0.10 / 1e6, "cost_out": 0.40 / 1e6
    },
    "gemini-2.0-flash": {
        "provider": "gemini", 
        "market_model": False, "benchmark_model": False, "is_judge": True,
        "cost_in": 0.10 / 1e6, "cost_out": 0.40 / 1e6
    },
    "deepseek-reasoner": { 
        "provider": "deepseek",
        "market_model": False, "benchmark_model": False, "is_judge": True,
        "cost_in": 0.28 / 1e6, "cost_out": 0.42 / 1e6,
        "tps": 25, "base_latency": 4.0
    },
    "gpt-5-mini-2025-08-07": {
        "provider": "openai",
        "market_model": False, "benchmark_model": True, "is_judge": False,
        "cost_in": 0.25 / 1e6, "cost_out": 2.00 / 1e6, "tps": 550, "base_latency": 0.25
    },
    
    # FREE === TIER 1: FAST === 
    "llama-3.1-8b-instant": {
        "provider": "groq",
        "market_model": False, "benchmark_model": True,
        "cost_in": 0.05 / 1e6, "cost_out": 0.08 / 1e6, "tps": 840, "base_latency": 0.2
    },
    "openai/gpt-oss-20b": {
        "provider": "groq",
        "market_model": True, "benchmark_model": False,
        "cost_in": 0.075 / 1e6, "cost_out": 0.30 / 1e6, "tps": 1000, "base_latency": 0.2
    },
    "meta-llama/llama-4-scout-17b-16e-instruct": {
        "provider": "groq", 
        "market_model": False, "benchmark_model": False, "is_classifier": True,
        "cost_in": 0.11 / 1e6, "cost_out": 0.34 / 1e6, "tps": 594, "base_latency": 0.25
    },
    "google/gemma-3n-e4b-it:free": { 
        "provider": "openrouter",
        "is_judge": False, "is_classifier": True,
        "cost_in": 0.02 / 1e6, "cost_out": 0.04 / 1e6, "base_latency": 0.14
        },
    "amazon/nova-2-lite-v1:free": {
        "provider": "openrouter", 
        "is_judge": False, "is_classifier": True,
        "cost_in": 0.30 / 1e6, "cost_out": 2.50 / 1e6, "base_latency": 0.62 
    },

    # FREE === TIER 2: MID RANGE ===
    "qwen/qwen3-32b": {
        "provider": "groq",
        "market_model": True, "benchmark_model": True,
        "cost_in": 0.29 / 1e6, "cost_out": 0.59 / 1e6, "tps": 662, "base_latency": 0.25
    },
    "openai/gpt-oss-120b": {
        "provider": "groq",
        "market_model": True, "benchmark_model": True,
        "cost_in": 0.15 / 1e6, "cost_out": 0.60 / 1e6, "tps": 500, "base_latency": 0.3
    },
    "llama-3.3-70b-versatile": {
        "provider": "groq",
        "market_model": False, "benchmark_model": False, "is_classifier": True,
        "cost_in": 0.59 / 1e6, "cost_out": 0.79 / 1e6, "tps": 300, "base_latency": 0.5
    },

    # FREE === TIER 3: SOTA ===
    "moonshotai/kimi-k2-instruct-0905": {
        "provider": "groq",
        "market_model": True, "benchmark_model": True,
        "cost_in": 1.00 / 1e6, "cost_out": 3.00 / 1e6, "tps": 200, "base_latency": 0.6
    },

    # PAID === TIER 1: FAST === 
    "gpt-5-nano-2025-08-07": {
        "provider": "openai",
        "market_model": True, "benchmark_model": False, "is_classifier": False,
        "cost_in": 0.05 / 1e6, "cost_out": 0.4 / 1e6, "tps": 750, "base_latency": 0.2
    },

    # PAID === TIER 3: SOTA ===
    "gpt-4.1-2025-04-14": {
        "provider": "openai",
        "market_model": False, "benchmark_model": True,
        "cost_in": 2.00 / 1e6, "cost_out": 8.00 / 1e6
    },
    "gemini-2.5-pro": { 
        "provider": "gemini",
        "market_model": False, "benchmark_model": False,
        "cost_in": 1.25 / 1e6, "cost_out": 10.00 / 1e6,
        "throttle_delay": 30 
    },
    "deepseek-chat": { 
        "provider": "deepseek",
        "market_model": True, "benchmark_model": True, "classifier_model": False,
        "cost_in": 0.28 / 1e6, "cost_out": 0.42 / 1e6,
        "tps": 70, "base_latency": 0.22
    },
}

# ---------------- FALLBACK ----------------
FALLBACK_CONFIG = {
    "provider": "groq",
    "model": "openai/gpt-oss-20b"
}

RETRY_CONFIG = {
    "max_retries": 3,
    "sleep_map": {"gemini": 30, "default": 2},
    "server_error_sleep": 5,
    "general_error_sleep": 2
}
MAX_API_RETRIES = 3