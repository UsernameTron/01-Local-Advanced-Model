import json
import time
from agents import create_ceo, create_executor, create_summarizer, create_test_generator, create_dependency_agent
from task_complexity_analyzer import analyze_task_complexity
from vector_memory import vector_memory

with open('orchestration_config.json') as f:
    config = json.load(f)

def run_advanced_pipeline(task, image_path=None, audio_path=None):
    total_start = time.time()
    complexity = analyze_task_complexity(task)
    template = config['workflow_templates'][complexity]
    max_executors = min(template['max_executors'], config['resource_limits']['max_total_executors'])
    parallel = template['parallel']
    agents_to_use = template['agents']
    print(f"[Orchestrator] Task complexity: {complexity}, Executors: {max_executors}, Parallel: {parallel}")

    multimodal_result = None
    if image_path:
        from agents import ImageAgent
        img_agent = ImageAgent()
        multimodal_result = img_agent.caption(image_path)
        vector_memory.add_image(image_path, multimodal_result)
    if audio_path:
        from agents import AudioAgent
        audio_agent = AudioAgent()
        audio_text = audio_agent.transcribe(audio_path)
        multimodal_result = audio_text
        vector_memory.add_audio(audio_path, audio_text)
    if multimodal_result:
        # Combine with text task if both present
        task = f"{task}\n[Image/Audio context]: {multimodal_result}"

    # Memory retrieval before agent execution
    cached = vector_memory.retrieve(task)
    if cached:
        print("[Memory] Cache hit. Returning cached output.")
        print(f"[Memory] Stats: {vector_memory.stats()}")
        return cached
    else:
        print("[Memory] Cache miss. Proceeding with agent execution.")

    ceo = create_ceo()
    plan = ceo.run(task)
    steps = [s for s in plan.split("\n") if s.strip()][:max_executors]

    results = []
    if parallel:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_executors) as executor:
            futures = [executor.submit(create_executor(i, complexity).run, step) for i, step in enumerate(steps)]
            for future in futures:
                results.append(future.result())
    else:
        for i, step in enumerate(steps):
            results.append(create_executor(i, complexity).run(step))

    if 'summarizer' in agents_to_use:
        summarizer = create_summarizer()
        summary = summarizer.run("\n".join(results))
        results.append(summary)
    if 'test_generator' in agents_to_use:
        test_agent = create_test_generator()
        test_results = [test_agent.generate_tests(step) for step in steps]
        results.extend(test_results)
    if 'dependency_agent' in agents_to_use:
        dep_agent = create_dependency_agent()
        dep_result = dep_agent.analyze_deps()
        results.append(dep_result)

    # Feedback loop: check quality and adapt
    if config['feedback']['enable_feedback_loop']:
        from validate_executor_quality import similarity
        quality_threshold = config['feedback']['quality_threshold']
        for i, output in enumerate(results):
            if isinstance(output, str) and len(output.strip()) < 10 and max_executors < config['resource_limits']['max_total_executors']:
                print(f"[Feedback] Output {i} too short, increasing executors and retrying...")
                return run_advanced_pipeline(task)

    # Persist to vector memory
    vector_memory.add(task, "\n".join(str(r) for r in results))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    with open(f"output/advanced_summary_{timestamp}.txt", "w") as f:
        f.write("\n".join(str(r) for r in results))
    print(f"Total advanced pipeline time: {time.time() - total_start:.2f}s")
    print(f"[Memory] Stats: {vector_memory.stats()}")
    return results

if __name__ == '__main__':
    import sys
    task = sys.argv[1] if len(sys.argv) > 1 else "Design a multi-agent research bootcamp pipeline with dynamic scaling."
    run_advanced_pipeline(task)
