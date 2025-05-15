from ollama import Client
from config import CEO_MODEL, FAST_MODEL, EXECUTOR_MODEL_ORIGINAL, EXECUTOR_MODEL_DISTILLED, USE_DISTILLED_EXECUTOR
import time
import torch
from transformers import CLIPProcessor, CLIPModel, ViTImageProcessor, ViTModel, WhisperProcessor, WhisperForConditionalGeneration
from PIL import Image
import numpy as np

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
    def generate_tests(self, module_path):
        return self.run(f"Generate pytest tests for {module_path}")

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

def create_summarizer(): return Agent('Summarizer', CEO_MODEL)
def create_test_generator(): return TestGeneratorAgent('TestGenerator', CEO_MODEL)
def create_dependency_agent(): return DependencyAgent('DependencyAgent', CEO_MODEL)
