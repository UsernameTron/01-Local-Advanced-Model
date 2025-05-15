# Advanced Orchestration for Local O1

## Overview
This module implements dynamic workflow scaling and adaptive orchestration for the Local O1 system, optimizing agent allocation and execution paths based on task complexity.

## Components
- **task_complexity_analyzer.py**: Classifies tasks as simple, medium, or complex.
- **orchestration_config.json**: Configures workflow templates, resource limits, and feedback loop.
- **advanced_orchestrator.py**: Runs the pipeline with dynamic scaling, parallelism, and adaptive routing.

## Workflow Templates
- **Simple**: 1 executor, sequential, summarizer only.
- **Medium**: Up to 3 executors, parallel, summarizer.
- **Complex**: Up to 5 executors, parallel, summarizer, test generator, dependency agent.

## Usage
```sh
python advanced_orchestrator.py "<your task description>"
```

## Configuration
- Edit `orchestration_config.json` to adjust workflow templates, resource limits, and feedback loop settings.

## Feedback Loop
- The orchestrator adapts executor allocation and retries if output quality is insufficient.

## Performance
- Compare before/after results using the benchmarking and reporting modules.

## Example Patterns
- Simple: "Summarize the main findings of the report."
- Medium: "Generate pytest tests for agents.py."
- Complex: "Design a multi-agent research bootcamp pipeline with dynamic scaling."

---

**This orchestration system is fully configurable and integrates with the optimization pipeline for continuous improvement.**
