import json
from typing import List

from google.auth import default as google_auth_default
from google.auth.transport.requests import AuthorizedSession

from conversationgenome.ConfigLib import c
from conversationgenome.llm.LlmLib import LlmLib


DEFAULT_MODEL = "gemini-3.6-flash"
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
REQUEST_TIMEOUT_SECONDS = 24
SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


class LlmVertex(LlmLib):
    def __init__(self, request_timeout: float | None = None):
        credentials, adc_project = google_auth_default(scopes=SCOPES)
        self.project = c.get("env", "GOOGLE_CLOUD_PROJECT", adc_project)
        if not self.project:
            raise ValueError(
                "GOOGLE_CLOUD_PROJECT environment variable not set and no project "
                "was found in Application Default Credentials."
            )

        self.location = c.get("env", "GOOGLE_CLOUD_LOCATION", "global")
        self.model = c.get("env", "VERTEX_MODEL", DEFAULT_MODEL)
        self.reasoning_effort = "LOW"
        self.embedding_model = c.get(
            "env", "VERTEX_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
        )
        self.request_timeout = (
            request_timeout if request_timeout is not None else REQUEST_TIMEOUT_SECONDS
        )
        self.client = AuthorizedSession(credentials)

    def _model_url(self, model: str, action: str) -> str:
        if self.location == "global":
            host = "aiplatform.googleapis.com"
        elif self.location in {"us", "eu"}:
            host = f"aiplatform.{self.location}.rep.googleapis.com"
        else:
            host = f"{self.location}-aiplatform.googleapis.com"
        return (
            f"https://{host}/v1/"
            f"projects/{self.project}/locations/{self.location}/"
            f"publishers/google/models/{model}:{action}"
        )

    def basic_prompt(self, prompt: str, response_format: str = "text") -> str | None:
        generation_config = {
            "thinkingConfig": {"thinkingLevel": self.reasoning_effort},
        }
        if response_format == "json":
            generation_config["responseMimeType"] = "application/json"

        attempts = 2 if response_format == "json" else 1
        for _ in range(attempts):
            response = None
            try:
                response = self.client.post(
                    self._model_url(self.model, "generateContent"),
                    json={
                        "contents": [
                            {"role": "user", "parts": [{"text": prompt}]}
                        ],
                        "generationConfig": generation_config,
                    },
                    timeout=self.request_timeout,
                )
                response.raise_for_status()
                parts = response.json()["candidates"][0]["content"]["parts"]
                content = "".join(part.get("text", "") for part in parts)
            except Exception as e:
                error_response = getattr(e, "response", None)
                if error_response is None:
                    error_response = response
                status = getattr(error_response, "status_code", None)
                body = (getattr(error_response, "text", "") or "").replace("\n", " ")
                print(
                    f"Vertex Completion Error: {type(e).__name__}: {e}; "
                    f"status={status}; body={body[:1000]}"
                )
                return None

            if response_format != "json":
                return content
            try:
                json.loads(content)
                return content
            except json.JSONDecodeError:
                pass

        print("Vertex JSON Error")
        return None

    def get_vector_embeddings(
        self, tag: str, dimensions: int = 1536
    ) -> List[float] | None:
        try:
            response = self.client.post(
                self._model_url(self.embedding_model, "predict"),
                json={
                    "instances": [
                        {
                            "content": tag.replace("\n", " "),
                            "task_type": "SEMANTIC_SIMILARITY",
                        }
                    ],
                    "parameters": {"outputDimensionality": dimensions},
                },
                timeout=self.request_timeout,
            )
            response.raise_for_status()
            return response.json()["predictions"][0]["embeddings"]["values"]
        except Exception:
            print("Vertex Embedding Error")
            return None
