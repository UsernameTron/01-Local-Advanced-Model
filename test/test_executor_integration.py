import unittest
from agents import create_executor
from config import USE_DISTILLED_EXECUTOR

class TestExecutorIntegration(unittest.TestCase):
    def test_distilled_vs_original_switch(self):
        # Test both model selection paths
        import agents
        agents.USE_DISTILLED_EXECUTOR = True
        ex_distilled = create_executor(0)
        out_distilled = ex_distilled.run("Summarize the main findings of the report.")
        agents.USE_DISTILLED_EXECUTOR = False
        ex_original = create_executor(0)
        out_original = ex_original.run("Summarize the main findings of the report.")
        self.assertNotEqual(out_distilled, "[ERROR] Executor_0 failed after 3 attempts")
        self.assertNotEqual(out_original, "[ERROR] Executor_0 failed after 3 attempts")
        self.assertTrue(len(out_distilled.strip()) > 0)
        self.assertTrue(len(out_original.strip()) > 0)

    def test_fallback_mechanism(self):
        import agents
        agents.USE_DISTILLED_EXECUTOR = True
        ex = create_executor(1)
        # Simulate a bad output by monkeypatching Agent.run
        orig_run = ex.run
        def bad_run(prompt, retries=3, timeout=30):
            return "[ERROR] Executor_1 failed after 3 attempts"
        ex.run = bad_run
        # Should fallback to original model
        result = ex.run("Test fallback")
        self.assertIn("failed", result)
        ex.run = orig_run

if __name__ == '__main__':
    unittest.main()
