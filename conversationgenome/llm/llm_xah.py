from concurrent.futures import FIRST_COMPLETED
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError
from concurrent.futures import wait
import json
import time

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
        self.primary_grace_seconds = float(
            c.get("env", "XAH_PRIMARY_GRACE_SECONDS", "8")
        )
        self.response_deadline_seconds = float(
            c.get("env", "XAH_RESPONSE_DEADLINE_SECONDS", "10")
        )
        self.long_response_deadline_seconds = float(
            c.get("env", "XAH_LONG_RESPONSE_DEADLINE_SECONDS", "20")
        )
        self.model = self.primary_model
        self.embedding_model = "text-embedding-3-small"

    def _request(self, model: str, prompt: str, response_format: str) -> str:
        params = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "timeout": self.response_deadline_seconds,
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
        deadline = time.monotonic() + self.response_deadline_seconds
        executor = ThreadPoolExecutor(max_workers=2)
        primary = executor.submit(
            self._request, self.primary_model, prompt, response_format
        )
        fallback = executor.submit(
            self._request, self.fallback_model, prompt, response_format
        )

        try:
            candidates = [(fallback, "fallback")]
            try:
                return primary.result(
                    timeout=min(
                        self.primary_grace_seconds,
                        max(0, deadline - time.monotonic()),
                    )
                )
            except TimeoutError:
                candidates.append((primary, "primary"))
            except Exception as primary_error:
                print(f"XAH primary model error: {primary_error}")

            pending = {future for future, _ in candidates if not future.done()}
            completed = {future for future, _ in candidates if future.done()}

            while completed or pending:
                for future, label in candidates:
                    if future not in completed:
                        continue
                    completed.remove(future)
                    try:
                        return future.result()
                    except Exception as error:
                        print(f"XAH {label} model error: {error}")

                if not pending:
                    return None

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break

                completed, pending = wait(
                    pending,
                    timeout=remaining,
                    return_when=FIRST_COMPLETED,
                )

                if not completed:
                    break

            print("XAH response deadline exceeded")
            return None
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
