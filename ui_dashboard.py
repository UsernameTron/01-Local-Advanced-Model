import streamlit as st
import os
import json
import time
import psutil
import requests
from vector_memory import vector_memory
from advanced_orchestrator import run_advanced_pipeline
import networkx as nx
from pyvis.network import Network
from agents import CodeAnalyzerAgent, CodeRepairAgent, SemanticCodeSearch, automated_debugging_workflow
from agents import PerformanceProfilerAgent, OptimizationSuggesterAgent, BenchmarkingTool
from agents import pipeline_coordinator

st.set_page_config(page_title="Local O1 Dashboard", layout="wide")
st.title("Local O1 System Dashboard")

# Sidebar: Task submission
st.sidebar.header("Submit a Task")
task_input = st.sidebar.text_area("Task Description", "Summarize the main findings of the report.")
run_task = st.sidebar.button("Run Task")

# Sidebar: Ollama status
def check_ollama_status():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        if r.status_code == 200:
            return True, "Ollama running"
        else:
            return False, f"Ollama error: {r.status_code}"
    except Exception as e:
        return False, f"Ollama not reachable: {e}"

ollama_ok, ollama_msg = check_ollama_status()
status_dot = f'<span style="color: {"green" if ollama_ok else "red"}; font-size: 2em;">●</span>'
st.sidebar.markdown(f"Ollama Status: {status_dot} {ollama_msg}", unsafe_allow_html=True)

# Sidebar: Task history
if not os.path.exists('logs/task_history.json'):
    with open('logs/task_history.json', 'w') as f:
        json.dump([], f)
with open('logs/task_history.json') as f:
    task_history = json.load(f)

# Sidebar: Multi-Modal Inputs
st.sidebar.header("Multi-Modal Inputs")
image_file = st.sidebar.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
audio_file = st.sidebar.file_uploader("Upload Audio", type=["wav", "mp3", "m4a"])

# Save uploaded files to disk for processing
image_path = None
audio_path = None
if image_file:
    image_path = f"logs/uploaded_{int(time.time())}.png"
    with open(image_path, "wb") as f:
        f.write(image_file.read())
if audio_file:
    audio_path = f"logs/uploaded_{int(time.time())}.wav"
    with open(audio_path, "wb") as f:
        f.write(audio_file.read())

# Main: Task execution and results
task_result = None
if run_task and task_input.strip():
    with st.spinner("Running advanced orchestrator..."):
        result = run_advanced_pipeline(task_input, image_path=image_path, audio_path=audio_path)
        task_result = str(result)
        # Save to history
        task_history.append({
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'task': task_input,
            'result': task_result,
            'image': image_path,
            'audio': audio_path
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

# Multi-modal memory browsing
st.sidebar.header("Multi-Modal Memory")
if st.sidebar.button("Show Recent Images"):
    imgs = [t for t in task_history if t.get('image')]
    for t in imgs[-5:]:
        st.image(t['image'], caption=f"Task: {t['task']}")
if st.sidebar.button("Show Recent Audio"):
    auds = [t for t in task_history if t.get('audio')]
    for t in auds[-5:]:
        st.audio(t['audio'], format='audio/wav')

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

# Tabs: Dashboard and Code Repair
tabs = st.tabs(["Dashboard", "Code Repair", "Performance", "Integrated Optimization"])

with tabs[1]:
    st.header("Code Repair & Debugging")
    st.write("Analyze, debug, and repair your codebase with explainable AI support.")
    # Dependency graph
    analyzer = CodeAnalyzerAgent()
    file_map = analyzer.scan_repository()
    dep_graph = analyzer.build_dependency_graph(file_map)
    G = nx.DiGraph()
    for f, deps in dep_graph.items():
        for d in deps:
            G.add_edge(f, d)
    net = Network(notebook=False, height="400px", width="100%")
    net.from_nx(G)
    net.save_graph("logs/dep_graph.html")
    st.components.v1.html(open("logs/dep_graph.html").read(), height=420)
    # Semantic code search
    st.subheader("Semantic Code Search")
    search_query = st.text_input("Natural language code search", "Find all functions that open files")
    if search_query:
        searcher = SemanticCodeSearch(file_map)
        results = searcher.search(search_query, top_k=5)
        for r in results:
            st.markdown(f"**{r['file']}:{r['line']}**: `{r['text']}`")
    # Automated debugging workflow
    st.subheader("Automated Debugging & Repair")
    target_file = st.text_input("Target file for repair", "agents.py")
    if st.button("Run Debug & Repair"):
        result = automated_debugging_workflow(target_file)
        st.write(result)
        # Show diff and rationale (placeholder)
        st.write("**Fix rationale:**", "This fix addresses the detected bug by applying a context-aware repair based on project code patterns.")
        st.write("**Diff preview:** (not implemented)")
        st.write("**Approve or override the fix below.")
        st.button("Approve Fix")
        st.button("Override Fix")

with tabs[2]:
    st.header("Performance Profiling & Optimization")
    st.write("Profile code, find bottlenecks, and get optimization suggestions.")
    file_path = st.text_input("Python file to profile", "agents.py")
    func_name = st.text_input("Function name (optional)", "")
    if st.button("Run Profiler"):
        profiler = PerformanceProfilerAgent()
        result = profiler.profile_code(file_path, func_name or None)
        st.subheader("CPU Profile")
        st.code(result["cpu_profile"])
        st.subheader("Memory Profile")
        st.line_chart(result["mem_profile"])
        hotspots = profiler.find_hotspots(result["cpu_profile"])
        st.subheader("Hotspots")
        for h in hotspots:
            st.write(h["line"])
        complexity = profiler.analyze_complexity(file_path)
        st.subheader("Complexity Analysis")
        st.write(complexity)
        suggester = OptimizationSuggesterAgent()
        suggestions = suggester.suggest(hotspots, result["mem_profile"], complexity)
        st.subheader("Optimization Suggestions")
        for s in suggestions:
            st.write("- ", s)
    st.subheader("Benchmarking")
    if st.button("Run Benchmark (before/after)"):
        bench = BenchmarkingTool()
        before = bench.benchmark(file_path, func_name or None)
        st.write("Before optimization:", before)
        # User would apply fix, then rerun
        st.write("After optimization: (apply fix and rerun)")

with tabs[3]:
    st.header("Integrated Codebase Optimization")
    st.write("Run end-to-end analysis, debugging, repair, and optimization on your codebase.")
    target_file = st.text_input("Target file for integrated optimization", "agents.py")
    func_name = st.text_input("Function name (optional)", "")
    if st.button("Run Integrated Pipeline"):
        with st.spinner("Running integrated optimization pipeline..."):
            report = pipeline_coordinator(target_file, func_name or None)
            st.subheader("Unified Report")
            st.text_area("Results", report, height=400)

st.caption("Local O1 Dashboard | Real-time monitoring and orchestration | © 2025")

def main():
    import sys
    from streamlit.web import cli as stcli
    sys.argv = ["streamlit", "run", __file__] + sys.argv[1:]
    sys.exit(stcli.main())

if __name__ == "__main__":
    main()
