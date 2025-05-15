import json
from difflib import SequenceMatcher

# Acceptance criteria: at least 90% similarity to original output, or passes custom checks
QUALITY_THRESHOLD = 0.9

def load_benchmark_results():
    default = {
        "latency": {"original": 1.0, "distilled": 0.8},
        "memory_usage": {"original": 512, "distilled": 256},
        "quality_score": {"original": 0.95, "distilled": 0.91},
        "original_outputs": ["Sample output from original model."],
        "distilled_outputs": ["Sample output from distilled model."]
    }
    try:
        with open('benchmark_results.json') as f:
            return json.load(f)
    except FileNotFoundError:
        print('[WARN] benchmark_results.json not found. Using default values.')
        return default
    except json.JSONDecodeError:
        print('[WARN] benchmark_results.json is malformed. Using default values.')
        return default

results = load_benchmark_results()

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

passed = 0
for d_out, o_out in zip(results['distilled_outputs'], results['original_outputs']):
    sim = similarity(d_out, o_out)
    print(f"Distilled vs. Original similarity: {sim:.2f}")
    if sim >= QUALITY_THRESHOLD:
        passed += 1
    else:
        print(f"[WARN] Output below threshold. Distilled: {d_out}\nOriginal: {o_out}")

print(f"\nQuality validation: {passed}/{len(results['distilled_outputs'])} outputs meet threshold ({QUALITY_THRESHOLD*100:.0f}%)")
if passed < len(results['distilled_outputs']):
    print("[ACTION] Fallback to original model recommended for some cases.")
else:
    print("All distilled outputs meet quality criteria.")
