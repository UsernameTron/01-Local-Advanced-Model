import streamlit as st
import os
import json
import time
import psutil
from vector_memory import vector_memory
from advanced_orchestrator import run_advanced_pipeline

st.set_page_config(page_title="Local O1 Dashboard", layout="wide")
st.title("Local O1 System Dashboard")

# Sidebar: Task submission
st.sidebar.header("Submit a Task")
task_input = st.sidebar.text_area("Task Description", "Summarize the main findings of the report.")
run_task = st.sidebar.button("Run Task")

# Sidebar: Task history
if not os.path.exists('logs/task_history.json'):
    with open('logs/task_history.json', 'w') as f:
        json.dump([], f)
with open('logs/task_history.json') as f:
    task_history = json.load(f)

# Main: Task execution and results
task_result = None
if run_task and task_input.strip():
    with st.spinner("Running advanced orchestrator..."):
        result = run_advanced_pipeline(task_input)
        task_result = str(result)
        # Save to history
        task_history.append({
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'task': task_input,
            'result': task_result
        })
        with open('logs/task_history.json', 'w') as f:
            json.dump(task_history, f, indent=2)

# Task history browsing
st.sidebar.header("Task History")
history_filter = st.sidebar.text_input("Filter by keyword")
filtered_history = [t for t in task_history if history_filter.lower() in t['task'].lower()]
st.sidebar.write(f"{len(filtered_history)} tasks found.")
for t in reversed(filtered_history[-10:]):
    st.sidebar.markdown(f"**{t['timestamp']}**\n- {t['task'][:60]}...\n- [Show Result](#)")
    if st.sidebar.button(f"Show Result {t['timestamp']}"):
        st.write(f"### Task: {t['task']}\n#### Result:\n{t['result']}")

# Main: Show last result
if task_result:
    st.subheader("Latest Task Result")
    st.code(task_result)

# Vector memory: show similar past tasks
if task_input.strip():
    st.subheader("Similar Past Tasks (Vector Memory)")
    similar = vector_memory.retrieve(task_input, top_k=3)
    if similar:
        st.write(similar)
    else:
        st.write("No similar tasks found in memory.")

# Monitoring: Agent performance, memory, system resources
st.header("System Monitoring")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Vector Memory Entries", vector_memory.stats()['entries'])
    st.metric("Cache Hits", vector_memory.stats()['hits'])
    st.metric("Cache Misses", vector_memory.stats()['misses'])
with col2:
    st.metric("Avg Retrieval Time (ms)", f"{vector_memory.stats()['avg_retrieval_time_ms']:.2f}")
    st.metric("Disk Usage (GB)", f"{vector_memory.stats()['disk_usage_gb']:.2f}")
with col3:
    st.metric("RAM Usage (GB)", f"{psutil.virtual_memory().used / (1024**3):.2f}")
    st.metric("CPU Usage (%)", f"{psutil.cpu_percent()}%")

# Visualization: Workflow execution (simple timeline)
if task_result:
    st.header("Workflow Execution Timeline")
    st.write("(For detailed agent timing, see logs or performance report.)")

# API endpoint (for programmatic interaction)
import streamlit.components.v1 as components
st.header("API Endpoint")
st.code("POST /run_task { 'task': <description> }")
st.write("(To implement: expose a REST API using FastAPI or Flask, or use Streamlit's experimental API support.)")

st.caption("Local O1 Dashboard | Real-time monitoring and orchestration | © 2025")
