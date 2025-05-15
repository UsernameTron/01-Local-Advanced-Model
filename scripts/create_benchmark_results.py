import os
import json

default_benchmark = {
    "latency": {"original": 1.0, "distilled": 0.8},
    "memory_usage": {"original": 512, "distilled": 256},
    "quality_score": {"original": 0.95, "distilled": 0.91},
    "original_outputs": ["Sample output from original model."],
    "distilled_outputs": ["Sample output from distilled model."]
}

if not os.path.exists("benchmark_results.json"):
    with open("benchmark_results.json", "w") as f:
        json.dump(default_benchmark, f, indent=2)
    print("Created default benchmark_results.json.")
else:
    print("benchmark_results.json already exists.")
