from ollama import Client
from config import CEO_MODEL, FAST_MODEL, EXECUTOR_MODEL_ORIGINAL, EXECUTOR_MODEL_DISTILLED, USE_DISTILLED_EXECUTOR
import time
import torch
from transformers import CLIPProcessor, CLIPModel, ViTImageProcessor, ViTModel, WhisperProcessor, WhisperForConditionalGeneration
from PIL import Image
import numpy as np
import os
import ast
import json
from vector_memory import vector_memory
from sentence_transformers import SentenceTransformer
import faiss
import re
import cProfile
import pstats
import io
import tokenize
from memory_profiler import memory_usage

class Agent:
    def __init__(self, name, model):
        self.name = name
        self.client = Client()
        self.model = model
    
    def run(self, prompt, retries=3, timeout=30):
        for attempt in range(retries):
            try:
                start_time = time.time()
                # Remove unsupported timeout argument
                response = self.client.chat(model=self.model, messages=[{"role": "user", "content": f"[{self.name}] {prompt}"}])
                elapsed = time.time() - start_time
                print(f"{self.name} response time: {elapsed:.2f}s")
                return response.message['content'] if hasattr(response, 'message') and 'content' in response.message else str(response)
            except Exception as e:
                print(f"[WARN] {self.name} failed (attempt {attempt+1}): {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        return f"[ERROR] {self.name} failed after {retries} attempts"

class TestGeneratorAgent(Agent):
    def generate_tests(self, module_path, bug_trace=None):
        base = self.run(f"Generate pytest tests for {module_path}")
        if bug_trace:
            # Extract function name and error line from bug_trace
            func_match = re.search(r'in (\w+)', bug_trace)
            func_name = func_match.group(1) if func_match else None
            line_match = re.search(r'File ".*", line (\d+)', bug_trace)
            line_no = int(line_match.group(1)) if line_match else None
            # Minimal failing test
            test_code = f"def test_{func_name or 'bug'}():\n    # Reproduces bug at line {line_no}\n    with pytest.raises(Exception):\n        {func_name or 'function'}()\n"
            return base + "\n# Minimal failing test generated:\n" + test_code
        return base

class DependencyAgent(Agent):
    def analyze_deps(self):
        return self.run("Analyze project dependencies and create requirements.txt")

def create_ceo(): return Agent('CEO', CEO_MODEL)
def create_executor(i, task_complexity=None):
    # Automatic model selection based on task complexity
    from config import EXECUTOR_MODEL_DISTILLED, EXECUTOR_MODEL_ORIGINAL, USE_DISTILLED_EXECUTOR
    if task_complexity == 'high':
        model = EXECUTOR_MODEL_ORIGINAL
    elif task_complexity == 'low':
        model = EXECUTOR_MODEL_DISTILLED
    else:
        model = EXECUTOR_MODEL_DISTILLED if USE_DISTILLED_EXECUTOR else EXECUTOR_MODEL_ORIGINAL
    return ExecutorWithFallback(f'Executor_{i}', model)

class ExecutorWithFallback(Agent):
    def run(self, prompt, retries=3, timeout=30):
        output = super().run(prompt, retries, timeout)
        # Quality check: fallback if output is error or too short
        if (output.startswith('[ERROR]') or len(output.strip()) < 10) and self.model == EXECUTOR_MODEL_DISTILLED:
            print(f"[FALLBACK] Distilled output failed, reverting to original model for {self.name}")
            self.model = EXECUTOR_MODEL_ORIGINAL
            return super().run(prompt, retries, timeout)
        return output

class ImageAgent:
    def __init__(self, device=None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    def embed_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            emb = self.model.get_image_features(**inputs)
        return emb.cpu().numpy().astype('float32')
    def caption(self, image_path, prompt="Describe this image."):
        # Placeholder: can be extended with BLIP or similar for captioning
        return "[Image captioning not implemented]"

class AudioAgent:
    def __init__(self, device=None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny").to(self.device)
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    def transcribe(self, audio_path):
        import torchaudio
        waveform, sr = torchaudio.load(audio_path)
        inputs = self.processor(waveform.squeeze().numpy(), sampling_rate=sr, return_tensors="pt").to(self.device)
        with torch.no_grad():
            predicted_ids = self.model.generate(inputs["input_features"])
        return self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

class CodeAnalyzerAgent:
    def __init__(self, root_dir="."):
        self.root_dir = root_dir
    def scan_repository(self):
        file_map = {}
        for dirpath, _, filenames in os.walk(self.root_dir):
            for fname in filenames:
                if fname.endswith(('.py', '.js', '.ts', '.java', '.cpp', '.c', '.go')):
                    fpath = os.path.join(dirpath, fname)
                    with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    file_map[fpath] = content
        return file_map
    def extract_imports(self, file_content):
        try:
            tree = ast.parse(file_content)
            return [n.module for n in ast.walk(tree) if isinstance(n, ast.Import)] + \
                   [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        except Exception:
            return []
    def build_dependency_graph(self, file_map):
        graph = {}
        for fpath, content in file_map.items():
            graph[fpath] = self.extract_imports(content)
        return graph
    def analyze(self):
        file_map = self.scan_repository()
        dep_graph = self.build_dependency_graph(file_map)
        vector_memory.add("codebase_dependency_graph", json.dumps(dep_graph))
        return dep_graph

class CodeDebuggerAgent:
    def __init__(self):
        pass
    def locate_bugs(self, file_content):
        # Placeholder: Use StaticAnalysisTool or linting
        import subprocess
        with open("/tmp/_debug.py", "w") as f:
            f.write(file_content)
        try:
            result = subprocess.run(["python3", "-m", "py_compile", "/tmp/_debug.py"], capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                return result.stderr
            return "No syntax errors detected."
        except Exception as e:
            return str(e)
    def trace_execution(self, file_content):
        # Placeholder: Could use sys.settrace or coverage
        return "[Execution tracing not implemented]"
    def identify_root_cause(self, diagnostics):
        # Placeholder: Use heuristics or LLM
        if "SyntaxError" in diagnostics:
            return "Syntax error detected."
        return "Root cause analysis not implemented."

class CodeRepairAgent:
    def __init__(self, file_map=None, embedding_index=None):
        self.file_map = file_map or {}
        self.embedding_index = embedding_index or CodeEmbeddingIndex()
    def generate_fix(self, file_content, diagnostics, bug_query=None):
        # RAG: retrieve top-K relevant code snippets
        context_snippets = []
        if bug_query and self.embedding_index.index:
            context_snippets = self.embedding_index.search(bug_query, top_k=3)
        context_text = '\n'.join([f"{s['file']}:{s['line']}: {s['text']}" for s in context_snippets])
        # Placeholder: Use LLM or pattern-based fix, with context
        fix_prompt = f"Bug: {diagnostics}\nContext:\n{context_text}\nCode:\n{file_content}\nSuggest a fix."
        # Here, you would call an LLM with fix_prompt; fallback to naive fix:
        if "SyntaxError" in diagnostics:
            lines = file_content.splitlines()
            for i, line in enumerate(lines):
                if "SyntaxError" in diagnostics and str(i+1) in diagnostics:
                    lines[i] = "# [AUTO-FIXED] " + line
            return "\n".join(lines)
        return file_content
    def test_solution(self, file_path):
        import subprocess
        try:
            result = subprocess.run(["python3", "-m", "py_compile", file_path], capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except Exception:
            return False
    def validate_repair(self, file_path):
        return self.test_solution(file_path)

class CodeEmbeddingIndex:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.file_snippets = []
    def build_index(self, file_map):
        self.file_snippets = []
        embeddings = []
        for fpath, content in file_map.items():
            for i, line in enumerate(content.splitlines()):
                snippet = {'file': fpath, 'line': i+1, 'text': line}
                self.file_snippets.append(snippet)
                embeddings.append(self.model.encode(line, convert_to_numpy=True))
        if embeddings:
            emb_matrix = np.vstack(embeddings).astype('float32')
            self.index = faiss.IndexFlatL2(emb_matrix.shape[1])
            self.index.add(emb_matrix)
    def search(self, query, top_k=5):
        if not self.index:
            return []
        q_emb = self.model.encode(query, convert_to_numpy=True).reshape(1, -1).astype('float32')
        D, I = self.index.search(q_emb, top_k)
        return [self.file_snippets[idx] for idx in I[0] if idx < len(self.file_snippets)]

# Semantic code search utility
class SemanticCodeSearch:
    def __init__(self, file_map, model_name='all-MiniLM-L6-v2'):
        self.file_map = file_map
        self.model = SentenceTransformer(model_name)
        self.snippets = []
        self.embeddings = []
        for fpath, content in file_map.items():
            for i, line in enumerate(content.splitlines()):
                self.snippets.append({'file': fpath, 'line': i+1, 'text': line})
                self.embeddings.append(self.model.encode(line, convert_to_numpy=True))
        if self.embeddings:
            self.emb_matrix = np.vstack(self.embeddings).astype('float32')
        else:
            self.emb_matrix = None
    def search(self, query, top_k=5):
        if self.emb_matrix is None:
            return []
        q_emb = self.model.encode(query, convert_to_numpy=True).reshape(1, -1).astype('float32')
        index = faiss.IndexFlatL2(self.emb_matrix.shape[1])
        index.add(self.emb_matrix)
        D, I = index.search(q_emb, top_k)
        return [self.snippets[idx] for idx in I[0] if idx < len(self.snippets)]

# Orchestration pattern for automated debugging workflow

def automated_debugging_workflow(target_file):
    analyzer = CodeAnalyzerAgent()
    debugger = CodeDebuggerAgent()
    repairer = CodeRepairAgent()
    # 1. Scan and analyze
    file_map = analyzer.scan_repository()
    dep_graph = analyzer.build_dependency_graph(file_map)
    # 2. Focus on target file
    content = file_map.get(target_file)
    if not content:
        return f"File {target_file} not found."
    diagnostics = debugger.locate_bugs(content)
    root_cause = debugger.identify_root_cause(diagnostics)
    # 3. Generate repair
    fixed_content = repairer.generate_fix(content, diagnostics)
    tmp_path = "/tmp/_repaired.py"
    with open(tmp_path, "w") as f:
        f.write(fixed_content)
    # 4. Test and validate
    test_passed = repairer.test_solution(tmp_path)
    vector_memory.add(f"repair_{target_file}", json.dumps({"diagnostics": diagnostics, "root_cause": root_cause, "test_passed": test_passed}))
    return {"diagnostics": diagnostics, "root_cause": root_cause, "test_passed": test_passed}

def create_summarizer(): return Agent('Summarizer', CEO_MODEL)
def create_test_generator(): return TestGeneratorAgent('TestGenerator', CEO_MODEL)
def create_dependency_agent(): return DependencyAgent('DependencyAgent', CEO_MODEL)

class PerformanceProfilerAgent:
    def __init__(self):
        pass
    def profile_code(self, file_path, func_name=None):
        pr = cProfile.Profile()
        stats_output = io.StringIO()
        # Dynamically import and run the function
        import importlib.util
        spec = importlib.util.spec_from_file_location("mod", file_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        target_func = getattr(mod, func_name) if func_name else None
        def run():
            if target_func:
                return target_func()
            elif hasattr(mod, "main"):
                return mod.main()
        pr.enable()
        result = run()
        pr.disable()
        ps = pstats.Stats(pr, stream=stats_output).sort_stats('cumulative')
        ps.print_stats(20)
        cpu_profile = stats_output.getvalue()
        # Memory profiling
        mem_profile = memory_usage((run,))
        return {"cpu_profile": cpu_profile, "mem_profile": mem_profile, "result": result}
    def find_hotspots(self, cpu_profile):
        lines = cpu_profile.splitlines()
        hotspots = []
        for line in lines:
            if line.strip().startswith(('[', 'ncalls')):
                continue
            parts = line.split()
            if len(parts) > 5:
                try:
                    time_spent = float(parts[3])
                    if time_spent > 0.01:  # threshold
                        hotspots.append({'line': line, 'time': time_spent})
                except Exception:
                    continue
        return sorted(hotspots, key=lambda x: -x['time'])[:5]
    def analyze_complexity(self, file_path):
        with open(file_path, 'r') as f:
            tokens = list(tokenize.generate_tokens(f.readline))
        func_complexities = {}
        func_name = None
        loop_stack = []
        for tok in tokens:
            if tok.type == tokenize.NAME and tok.string == 'def':
                func_name = None
            elif tok.type == tokenize.NAME and func_name is None:
                func_name = tok.string
                func_complexities[func_name] = 1
            elif tok.type == tokenize.NAME and tok.string in ('for', 'while') and func_name:
                func_complexities[func_name] *= 10  # crude: each loop increases complexity
        # Heuristic: >100 means likely O(n^2) or worse
        return {k: ('O(n^2) or worse' if v > 100 else 'O(n) or better') for k, v in func_complexities.items()}

class OptimizationSuggesterAgent:
    def __init__(self):
        pass
    def suggest(self, cpu_profile, mem_profile, complexity_report):
        suggestions = []
        for func, complexity in complexity_report.items():
            if 'O(n^2)' in complexity:
                suggestions.append(f"Function {func} has high complexity: {complexity}. Consider optimizing nested loops.")
        for hotspot in cpu_profile:
            suggestions.append(f"Hotspot: {hotspot['line']} (time: {hotspot['time']:.4f}s). Consider refactoring or memoization.")
        if max(mem_profile) - min(mem_profile) > 100:
            suggestions.append("High memory usage detected. Check for leaks or large data structures.")
        return suggestions

# Benchmarking framework
class BenchmarkingTool:
    def __init__(self):
        pass
    def benchmark(self, file_path, func_name=None, repeat=3):
        import timeit
        import importlib.util
        spec = importlib.util.spec_from_file_location("mod", file_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        target_func = getattr(mod, func_name) if func_name else None
        stmt = f"mod.{func_name}()" if func_name else "mod.main()"
        setup = f"from __main__ import mod"
        times = timeit.repeat(stmt=stmt, setup=setup, repeat=repeat, number=1, globals={'mod': mod})
        return {'min': min(times), 'max': max(times), 'avg': sum(times)/len(times)}

class IntegratedCodebaseOptimizer:
    def __init__(self, root_dir="."):
        self.analyzer = CodeAnalyzerAgent(root_dir)
        self.debugger = CodeDebuggerAgent()
        self.repairer = CodeRepairAgent()
        self.profiler = PerformanceProfilerAgent()
        self.suggester = OptimizationSuggesterAgent()
        self.benchmarker = BenchmarkingTool()
    def optimize(self, target_file, func_name=None):
        # 1. Analyze codebase and build dependency graph
        file_map = self.analyzer.scan_repository()
        dep_graph = self.analyzer.build_dependency_graph(file_map)
        # 2. Prioritize high-impact components (most dependencies)
        impact_scores = {f: len(deps) for f, deps in dep_graph.items()}
        prioritized = sorted(impact_scores, key=impact_scores.get, reverse=True)
        report = {"fixed_bugs": [], "performance": {}}
        for f in prioritized:
            content = file_map.get(f)
            if not content:
                continue
            # 3. Debug and repair
            diagnostics = self.debugger.locate_bugs(content)
            if "SyntaxError" in diagnostics or "Exception" in diagnostics:
                root_cause = self.debugger.identify_root_cause(diagnostics)
                fixed_content = self.repairer.generate_fix(content, diagnostics)
                tmp_path = "/tmp/_repaired.py"
                with open(tmp_path, "w") as out:
                    out.write(fixed_content)
                test_passed = self.repairer.test_solution(tmp_path)
                report["fixed_bugs"].append({"file": f, "diagnostics": diagnostics, "root_cause": root_cause, "test_passed": test_passed})
                # 4. Profile and optimize
                orig_prof = self.profiler.profile_code(f, func_name)
                orig_hotspots = self.profiler.find_hotspots(orig_prof["cpu_profile"])
                orig_complexity = self.profiler.analyze_complexity(f)
                opt_prof = self.profiler.profile_code(tmp_path, func_name)
                opt_hotspots = self.profiler.find_hotspots(opt_prof["cpu_profile"])
                opt_complexity = self.profiler.analyze_complexity(tmp_path)
                suggestions = self.suggester.suggest(opt_hotspots, opt_prof["mem_profile"], opt_complexity)
                # 5. Benchmark A/B
                orig_bench = self.benchmarker.benchmark(f, func_name)
                opt_bench = self.benchmarker.benchmark(tmp_path, func_name)
                report["performance"][f] = {
                    "original": {"profile": orig_prof, "bench": orig_bench},
                    "optimized": {"profile": opt_prof, "bench": opt_bench},
                    "suggestions": suggestions
                }
        return report

def pipeline_coordinator(target_file, func_name=None, root_dir="."):
    optimizer = IntegratedCodebaseOptimizer(root_dir)
    report = optimizer.optimize(target_file, func_name)
    # Unified reporting
    summary = []
    for bug in report["fixed_bugs"]:
        summary.append(f"Fixed bug in {bug['file']}: {bug['diagnostics']} (root cause: {bug['root_cause']}, test passed: {bug['test_passed']})")
    for f, perf in report["performance"].items():
        summary.append(f"Performance for {f}:\n  Original: {perf['original']['bench']}\n  Optimized: {perf['optimized']['bench']}\n  Suggestions: {perf['suggestions']}")
    return "\n".join(summary)
