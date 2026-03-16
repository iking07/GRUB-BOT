import os
from litellm import completion
from .base import BaseProvider

class GroqProvider(BaseProvider):
    def __init__(self, model: str = "groq/llama-3.3-70b-versatile"):
        self.model = model
        
    def generate(self, prompt: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set. Add it to your environment or .env file.")

        response = completion(
            model=self.model,
            messages=messages,
            api_key=api_key,
        )
        return response.choices[0].message.content
