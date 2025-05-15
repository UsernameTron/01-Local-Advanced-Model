import re

def analyze_task_complexity(task: str) -> str:
    """
    Analyze the incoming task and return 'simple', 'medium', or 'complex'.
    """
    # Heuristic: count keywords, length, and structure
    keywords_complex = ["strategy", "pipeline", "architecture", "multi-agent", "research", "bootcamp", "launch", "dependency", "summarize", "plan", "analyze"]
    keywords_simple = ["test", "summarize", "list", "find", "describe", "generate"]
    task_lower = task.lower()
    score = 0
    for kw in keywords_complex:
        if kw in task_lower:
            score += 2
    for kw in keywords_simple:
        if kw in task_lower:
            score -= 1
    if len(task) > 200 or score > 3:
        return 'complex'
    elif len(task) > 80 or score > 1:
        return 'medium'
    else:
        return 'simple'

if __name__ == '__main__':
    # Example usage
    for t in [
        "Summarize the main findings of the report.",
        "Generate pytest tests for agents.py.",
        "Design a multi-agent research bootcamp pipeline with dynamic scaling.",
        "Analyze project dependencies and create requirements.txt."
    ]:
        print(f"Task: {t}\nComplexity: {analyze_task_complexity(t)}\n")
