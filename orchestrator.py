from agents import create_ceo, create_executor, create_summarizer
from datetime import datetime
import os
import time
from concurrent.futures import ThreadPoolExecutor
from prompts import CEO_PROMPT

def run_executors_parallel(steps):
    results = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(create_executor(i).run, step)
                  for i, step in enumerate(steps) if step.strip()]
        for future in futures:
            results.append(future.result())
    return results

def run_pipeline(task):
    total_start = time.time()
    ceo = create_ceo()
    # Use tuned CEO prompt to limit steps
    plan = ceo.run(f"{CEO_PROMPT.replace('{task}', task)}")
    steps = [s for s in plan.split("\n") if s.strip()][:5]  # Enforce max 5 steps
    results = run_executors_parallel(steps)
    summarizer = create_summarizer()
    final = summarizer.run("\n".join(results))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    with open(f"output/summary_{timestamp}.txt", "w") as f:
        f.write(final)
    print(f"Total pipeline time: {time.time() - total_start:.2f}s")

def run_code_quality_pipeline():
    from agents import create_test_generator, create_dependency_agent
    test_agent = create_test_generator()
    dep_agent = create_dependency_agent()

    # Generate tests for each module
    for module in ['agents.py', 'orchestrator.py', 'prompts.py']:
        tests = test_agent.generate_tests(module)
        with open(f"test/test_{module}", "w") as f:
            f.write(tests)

    # Generate requirements
    requirements = dep_agent.analyze_deps()
    with open("requirements.txt", "w") as f:
        f.write(requirements)

if __name__ == '__main__':
    run_pipeline("Design a five-day AI research bootcamp")
    run_code_quality_pipeline()
