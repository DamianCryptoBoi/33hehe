from concurrent.futures import ThreadPoolExecutor
import json

from openai import OpenAI

from conversationgenome.ConfigLib import c
from conversationgenome.llm.llm_openai import LlmOpenAI


class LlmXah(LlmOpenAI):
    def __init__(self):
        api_key = c.get("env", "XAH_API_KEY")
        if not api_key:
            raise ValueError("XAH_API_KEY environment variable not set.")

        self.client = OpenAI(
            api_key=api_key,
            base_url=c.get("env", "XAH_BASE_URL", "https://api.xah.io/v1"),
        )
        self.primary_model = c.get("env", "XAH_PRIMARY_MODEL", "deepseek-v4-flash")
        self.fallback_model = c.get(
            "env",
            "XAH_FALLBACK_MODEL",
            "levuphong2909/gemini-3.5-flash-high",
        )
        self.model = self.primary_model
        self.embedding_model = "text-embedding-3-small"

    def _request(self, model: str, prompt: str, response_format: str) -> str:
        params = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if response_format == "json":
            params["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**params)
        content = response.choices[0].message.content or ""
        if not content:
            raise ValueError(f"{model} returned an empty response")
        if response_format == "json":
            json.loads(content)
        return content

    def basic_prompt(self, prompt: str, response_format: str = "text") -> str | None:
        executor = ThreadPoolExecutor(max_workers=2)
        primary = executor.submit(
            self._request, self.primary_model, prompt, response_format
        )
        fallback = executor.submit(
            self._request, self.fallback_model, prompt, response_format
        )

        try:
            try:
                return primary.result()
            except Exception as primary_error:
                print(f"XAH primary model error: {primary_error}")
                try:
                    return fallback.result()
                except Exception as fallback_error:
                    print(f"XAH fallback model error: {fallback_error}")
                    return None
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
