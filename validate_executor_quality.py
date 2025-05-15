import json
from difflib import SequenceMatcher

# Acceptance criteria: at least 90% similarity to original output, or passes custom checks
QUALITY_THRESHOLD = 0.9

with open('benchmark_results.json') as f:
    results = json.load(f)

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
