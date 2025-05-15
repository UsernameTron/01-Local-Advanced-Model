import os
import sys
import json
import time
import importlib
import traceback
from datetime import datetime

RESULTS_FILE = "system_test_results.json"

class SystemTester:
    def __init__(self):
        self.passed = []
        self.failed = []
        self.summary = []
        self._ensure_dirs()

    def _ensure_dirs(self):
        for d in ["logs", "output", "dataset", "test"]:
            os.makedirs(d, exist_ok=True)

    def test_python_dependencies(self):
        try:
            import faiss
            import transformers
            import streamlit
            import sentence_transformers
            import pyvis
            import networkx
            import psutil
            import requests
            self.passed.append("python_dependencies")
        except Exception as e:
            self.failed.append(("python_dependencies", traceback.format_exc()))

    def test_ollama_status(self):
        try:
            import requests
            r = requests.get("http://localhost:11434/api/tags", timeout=2)
            if r.status_code == 200:
                self.passed.append("ollama_status")
            else:
                self.failed.append(("ollama_status", f"Status code: {r.status_code}"))
        except Exception as e:
            self.failed.append(("ollama_status", traceback.format_exc()))

    def test_agent_construction(self):
        try:
            from agents import CodeAnalyzerAgent, create_executor, create_ceo
            a = CodeAnalyzerAgent()
            e = create_executor(0)
            c = create_ceo()
            self.passed.append("agent_construction")
        except Exception as e:
            self.failed.append(("agent_construction", traceback.format_exc()))

    def test_vector_memory(self):
        try:
            from vector_memory import VectorMemory
            vm = VectorMemory(index_file="test_memory.index", meta_file="test_memory_meta.json")
            vm.add("test", "output", meta={"test": True})
            assert vm.retrieve("test") is not None
            os.remove("test_memory.index")
            os.remove("test_memory_meta.json")
            self.passed.append("vector_memory")
        except Exception as e:
            self.failed.append(("vector_memory", traceback.format_exc()))

    def test_orchestrator_mock(self):
        try:
            from advanced_orchestrator import run_advanced_pipeline
            # Use a mock task and skip live inference
            result = run_advanced_pipeline("Mock test task", image_path=None, audio_path=None)
            self.passed.append("orchestrator_mock")
        except Exception as e:
            self.failed.append(("orchestrator_mock", traceback.format_exc()))

    def test_code_intelligence(self):
        try:
            from agents import CodeAnalyzerAgent, PerformanceProfilerAgent
            # Create a minimal test file
            code = "def foo():\n return 42\n"
            with open("test_code.py", "w") as f:
                f.write(code)
            analyzer = CodeAnalyzerAgent()
            file_map = analyzer.scan_repository()
            profiler = PerformanceProfilerAgent()
            profiler.profile_code("test_code.py", "foo")
            os.remove("test_code.py")
            self.passed.append("code_intelligence")
        except Exception as e:
            self.failed.append(("code_intelligence", traceback.format_exc()))

    def test_dashboard_files(self):
        try:
            assert os.path.exists("ui_dashboard.py")
            import streamlit
            self.passed.append("dashboard_files")
        except Exception as e:
            self.failed.append(("dashboard_files", traceback.format_exc()))

    def run_all(self):
        self.test_python_dependencies()
        self.test_ollama_status()
        self.test_agent_construction()
        self.test_vector_memory()
        self.test_orchestrator_mock()
        self.test_code_intelligence()
        self.test_dashboard_files()
        self._report()

    def _report(self):
        print("\n===== SYSTEM TEST SUMMARY =====")
        print(f"Passed: {self.passed}")
        if self.failed:
            print(f"Failed: {[f[0] for f in self.failed]}")
            for name, err in self.failed:
                print(f"--- {name} ---\n{err}\n")
        else:
            print("All tests passed!")
        results = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "results": {
                "passed": self.passed,
                "failed": [{"name": n, "trace": t} for n, t in self.failed]
            }
        }
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    tester = SystemTester()
    tester.run_all()
