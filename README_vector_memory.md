# Vector Memory for Local O1

## Overview
This module provides persistent, vector-based memory for agent outputs, reducing redundant reasoning and improving efficiency.

## Features
- **Semantic retrieval**: Uses FAISS and MiniLM embeddings for fast, accurate similarity search.
- **Persistence**: Stores memory on disk with automatic pruning to respect RAM/disk constraints.
- **Cache tracking**: Logs hits, misses, and retrieval latency for effectiveness analysis.
- **Privacy controls**: Supports marking entries as sensitive (not persisted/exported).
- **Integration**: Hooks into the advanced orchestrator for pre-execution retrieval and post-execution storage.

## Usage
- Memory is checked before agent execution; if a similar result exists, it is returned immediately.
- New results are added to memory after execution.
- Pruning and stats are managed automatically.

## Configuration
- Vector dimension, disk/RAM limits, and pruning age are set in `vector_memory.py`.

## Example
```python
from vector_memory import vector_memory
cached = vector_memory.retrieve("Summarize the main findings of the report.")
if cached:
    print("Cache hit!", cached)
else:
    # Run agent, then store
    vector_memory.add(prompt, output)
```

## Performance
- Retrieval latency: <100ms
- Disk usage: <10GB (auto-pruned)
- RAM usage: <48GB (auto-pruned)

---

**This memory system is designed for high efficiency, privacy, and seamless integration with Local O1 orchestration.**
