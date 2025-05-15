PROMPT_SYSTEM = "You are a concise AI expert. Respond with minimal tokens."

PROMPT_LAUNCH = {
    "system": PROMPT_SYSTEM,
    "user": "Deconstruct a five-phase launch strategy for a biosensor startup."
}

CEO_PROMPT = "Break this task into 3-5 discrete, high-impact steps. Each step must be under 100 tokens. Return only the steps as a numbered list."

EXECUTOR_PROMPT = "For the following step, provide a focused, actionable response in under 100 tokens."

TEST_PROMPT = """
You are a test generation specialist. Create comprehensive pytest test cases for:
{module_path}

Focus on:
1. Testing all public methods
2. Edge cases and error handling
3. Mocking external dependencies

Return only valid Python code.
"""

DEPENDENCY_PROMPT = """
You are a dependency management specialist. Analyze this project structure
and create a requirements.txt file with:

1. All direct dependencies with version pins
2. Optional dev dependencies in a commented section
3. Security best practices

Return only the requirements.txt content.
"""

IMAGE_CAPTION_PROMPT = "Describe the content and context of this image."
AUDIO_TRANSCRIBE_PROMPT = "Transcribe the following audio and summarize its main points."
MULTIMODAL_REASONING_PROMPT = "Given the following text, image, and audio, provide a comprehensive analysis."
