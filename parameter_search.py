import json
import itertools
from agents import create_executor
from config import USE_DISTILLED_EXECUTOR

# Parameter grid (example: adjust as needed)
PARAM_GRID = {
    'USE_DISTILLED_EXECUTOR': [True, False],
    'MAX_EXECUTOR_STEPS': [3, 5],
    'TEMPERATURE': [0.5, 0.7, 1.0],
    'TOP_P': [0.8, 0.95],
}

BENCHMARK_STEPS = [
    "Analyze the requirements for a new authentication module.",
    "Design a database schema for user management.",
    "Implement error handling for API endpoints.",
    "Generate pytest tests for agents.py.",
    "Analyze project dependencies and create requirements.txt."
]


def run_benchmark(params):
    # Patch config globals
    import config
    for k, v in params.items():
        setattr(config, k, v)
    import agents
    agents.USE_DISTILLED_EXECUTOR = params['USE_DISTILLED_EXECUTOR']
    times = []
    for i, step in enumerate(BENCHMARK_STEPS[:config.MAX_EXECUTOR_STEPS]):
        agent = create_executor(i)
        import time
        start = time.time()
        agent.run(step)
        elapsed = time.time() - start
        times.append(elapsed)
    return sum(times) / len(times)


def main():
    keys, values = zip(*PARAM_GRID.items())
    best_score = float('inf')
    best_params = None
    results = []
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        avg_time = run_benchmark(params)
        results.append({'params': params, 'avg_time': avg_time})
        print(f"Params: {params} -> Avg Time: {avg_time:.2f}s")
        if avg_time < best_score:
            best_score = avg_time
            best_params = params
    with open('parameter_search_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Best params: {best_params} (Avg Time: {best_score:.2f}s)")
    with open('best_params.json', 'w') as f:
        json.dump(best_params, f, indent=2)

if __name__ == '__main__':
    main()
