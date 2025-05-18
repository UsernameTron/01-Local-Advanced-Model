import psutil
import threading
import time

class OptimizedMemoryManager:
    def __init__(self, max_ram_gb=48, compress_threshold=0.9):
        self.max_ram_bytes = max_ram_gb * 1024 ** 3
        self.compress_threshold = compress_threshold
        self.monitor_thread = threading.Thread(target=self._monitor, daemon=True)
        self.running = False

    def enable_auto_compression(self):
        self.running = True
        self.monitor_thread.start()

    def _monitor(self):
        while self.running:
            used = psutil.virtual_memory().used
            if used > self.max_ram_bytes * self.compress_threshold:
                self._compress_memory()
            time.sleep(2)

    def _compress_memory(self):
        print("[MEMORY MANAGER] High RAM usage detected. Triggering memory compression/GC.")
        import gc
        gc.collect()
        # Placeholder: add model-specific or data compression logic here

    def stop(self):
        self.running = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join()
