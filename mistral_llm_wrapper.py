from dotenv import load_dotenv
import os
from mistralai import Mistral
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any

load_dotenv()

class MistralChat(LLM):
    model: str = "mistral-large-latest"  # This is okay for pydantic

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY environment variable not set")

        client = Mistral(api_key=api_key)
        messages = [{"role": "user", "content": prompt}]
        response = client.chat.complete(model=self.model, messages=messages)
        return response.choices[0].message.content

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model": self.model}

    @property
    def _llm_type(self) -> str:
        return "mistral_chat"
