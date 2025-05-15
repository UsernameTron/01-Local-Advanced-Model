import time
from agents import create_executor
from config import USE_DISTILLED_EXECUTOR

BENCHMARK_STEPS = [
    "Analyze the requirements for a new authentication module.",
    "Design a database schema for user management.",
    "Implement error handling for API endpoints.",
    "Generate pytest tests for agents.py.",
    "Analyze project dependencies and create requirements.txt."
]


def benchmark_executor(distilled: bool):
    from config import EXECUTOR_MODEL_DISTILLED, EXECUTOR_MODEL_ORIGINAL
    import agents
    # Patch config for this run
    agents.USE_DISTILLED_EXECUTOR = distilled
    times = []
    outputs = []
    for i, step in enumerate(BENCHMARK_STEPS):
        agent = create_executor(i)
        start = time.time()
        output = agent.run(step)
        elapsed = time.time() - start
        times.append(elapsed)
        outputs.append(output)
        print(f"Executor_{i} ({'distilled' if distilled else 'original'}) response time: {elapsed:.2f}s")
    return times, outputs

if __name__ == '__main__':
    print("Benchmarking distilled executor...")
    t_distilled, out_distilled = benchmark_executor(True)
    print("\nBenchmarking original executor...")
    t_original, out_original = benchmark_executor(False)
    print("\n--- Results ---")
    print(f"Distilled avg: {sum(t_distilled)/len(t_distilled):.2f}s, Original avg: {sum(t_original)/len(t_original):.2f}s")
    print(f"Performance improvement: {100*(sum(t_original)-sum(t_distilled))/sum(t_original):.1f}%")
    # Save results
    with open('benchmark_results.json', 'w') as f:
        import json
        json.dump({
            'distilled_times': t_distilled,
            'original_times': t_original,
            'distilled_outputs': out_distilled,
            'original_outputs': out_original
        }, f, indent=2)
