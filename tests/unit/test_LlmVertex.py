import importlib
import importlib.util

import pytest

from conversationgenome.api.models.skill_coverage import SectionMapEntry


class FakeResponse:
    def __init__(self, body=None, error=None):
        self.body = body or {}
        self.error = error

    def raise_for_status(self):
        if self.error:
            raise self.error

    def json(self):
        return self.body


class FakeSession:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.posts = []

    def post(self, url, **kwargs):
        self.posts.append((url, kwargs))
        return self.responses.pop(0)


def load_vertex_module():
    module_name = "conversationgenome.llm.llm_vertex"
    assert importlib.util.find_spec(module_name) is not None, "Vertex backend is missing"
    return importlib.import_module(module_name)


def configure_vertex(monkeypatch, module, session, overrides=None):
    values = {
        ("env", "GOOGLE_CLOUD_PROJECT"): "test-project",
        ("env", "GOOGLE_CLOUD_LOCATION"): "global",
    }
    values.update(overrides or {})

    monkeypatch.setattr(
        module.c,
        "get",
        lambda section, key, default=None: values.get((section, key), default),
    )
    monkeypatch.setattr(
        module,
        "google_auth_default",
        lambda **kwargs: (object(), "adc-project"),
    )
    monkeypatch.setattr(module, "AuthorizedSession", lambda credentials: session)


def test_adc_project_scopes_and_credentials_are_used(monkeypatch):
    module = load_vertex_module()
    credentials = object()
    captured = {}
    monkeypatch.setattr(
        module.c,
        "get",
        lambda section, key, default=None: default,
    )

    def fake_default(**kwargs):
        captured["scopes"] = kwargs["scopes"]
        return credentials, "adc-project"

    def fake_session(received_credentials):
        captured["credentials"] = received_credentials
        return FakeSession()

    monkeypatch.setattr(module, "google_auth_default", fake_default)
    monkeypatch.setattr(module, "AuthorizedSession", fake_session)

    llm = module.LlmVertex()

    assert llm.project == "adc-project"
    assert captured == {
        "scopes": ["https://www.googleapis.com/auth/cloud-platform"],
        "credentials": credentials,
    }


def test_basic_prompt_uses_adc_and_minimal_thinking(monkeypatch):
    module = load_vertex_module()
    response = FakeResponse(
        {
            "candidates": [
                {"content": {"parts": [{"text": "alpha,beta"}]}}
            ]
        }
    )
    session = FakeSession(response)
    configure_vertex(monkeypatch, module, session)

    llm = module.LlmVertex()

    assert llm.basic_prompt("Tag this conversation") == "alpha,beta"
    assert session.posts == [
        (
            "https://aiplatform.googleapis.com/v1/projects/test-project/locations/global/"
            "publishers/google/models/gemini-3.6-flash:generateContent",
            {
                "json": {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [{"text": "Tag this conversation"}],
                        }
                    ],
                    "generationConfig": {
                        "thinkingConfig": {"thinkingLevel": "MINIMAL"}
                    },
                },
                "timeout": 24,
            },
        )
    ]


@pytest.mark.parametrize(
    ("location", "host"),
    [
        ("global", "aiplatform.googleapis.com"),
        ("us", "aiplatform.us.rep.googleapis.com"),
        ("eu", "aiplatform.eu.rep.googleapis.com"),
        ("us-central1", "us-central1-aiplatform.googleapis.com"),
    ],
)
def test_model_url_uses_location_specific_endpoint(monkeypatch, location, host):
    module = load_vertex_module()
    configure_vertex(
        monkeypatch,
        module,
        FakeSession(),
        overrides={("env", "GOOGLE_CLOUD_LOCATION"): location},
    )

    llm = module.LlmVertex()

    assert llm._model_url("gemini-3.6-flash", "generateContent") == (
        f"https://{host}/v1/projects/test-project/locations/{location}/"
        "publishers/google/models/gemini-3.6-flash:generateContent"
    )


def test_skill_coverage_prompt_uses_low_thinking(monkeypatch):
    module = load_vertex_module()
    session = FakeSession(
        FakeResponse(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": (
                                        '{"skill":"# Skill","tdd_plan":"Plan",'
                                        '"section_tests":{"s1":[{"name":"test",'
                                        '"description":"description","assertion":"assertion"}]}}'
                                    )
                                }
                            ]
                        }
                    }
                ]
            }
        ),
        FakeResponse(
            {"candidates": [{"content": {"parts": [{"text": "alpha,beta"}]}}]}
        ),
    )
    configure_vertex(monkeypatch, module, session)
    llm = module.LlmVertex()

    result = llm.skill_request_to_skill_bundle(
        "Build a skill",
        [SectionMapEntry(section_id="s1", title="Section", description="Do it")],
    )

    assert result.skill == "# Skill"
    assert session.posts[0][1]["json"]["generationConfig"]["thinkingConfig"] == {
        "thinkingLevel": "LOW"
    }
    assert llm.basic_prompt("Tag this conversation") == "alpha,beta"
    assert session.posts[1][1]["json"]["generationConfig"]["thinkingConfig"] == {
        "thinkingLevel": "MINIMAL"
    }


def test_fallback_skill_coverage_prompts_use_low_thinking(monkeypatch):
    module = load_vertex_module()
    session = FakeSession(
        FakeResponse(
            {"candidates": [{"content": {"parts": [{"text": "# Skill"}]}}]}
        ),
        FakeResponse(
            {"candidates": [{"content": {"parts": [{"text": "Plan"}]}}]}
        ),
        FakeResponse(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": (
                                        '{"section_tests":{"s1":[{"name":"test",'
                                        '"description":"description","assertion":"assertion"}]}}'
                                    )
                                }
                            ]
                        }
                    }
                ]
            }
        ),
    )
    configure_vertex(monkeypatch, module, session)
    llm = module.LlmVertex()
    section_map = [
        SectionMapEntry(section_id="s1", title="Section", description="Do it")
    ]

    assert llm.skill_request_to_skill("Build a skill", section_map) == "# Skill"
    assert llm.skill_to_tdd_plan("# Skill", section_map) == "Plan"
    assert llm.skill_to_section_tests("# Skill", "Plan", section_map).success

    assert [
        post[1]["json"]["generationConfig"]["thinkingConfig"]
        for post in session.posts
    ] == [{"thinkingLevel": "LOW"}] * 3


def test_skill_coverage_thinking_is_restored_after_error(monkeypatch):
    module = load_vertex_module()
    session = FakeSession(
        FakeResponse(error=RuntimeError("unavailable")),
        FakeResponse(
            {"candidates": [{"content": {"parts": [{"text": "alpha,beta"}]}}]}
        ),
    )
    configure_vertex(monkeypatch, module, session)
    llm = module.LlmVertex()
    section_map = [
        SectionMapEntry(section_id="s1", title="Section", description="Do it")
    ]

    assert llm.skill_request_to_skill_bundle("Build a skill", section_map) is None
    assert llm.basic_prompt("Tag this conversation") == "alpha,beta"
    assert [
        post[1]["json"]["generationConfig"]["thinkingConfig"]
        for post in session.posts
    ] == [{"thinkingLevel": "LOW"}, {"thinkingLevel": "MINIMAL"}]


def test_json_prompt_requests_json_response(monkeypatch):
    module = load_vertex_module()
    session = FakeSession(
        FakeResponse(
            {"candidates": [{"content": {"parts": [{"text": '{"ok":true}'}]}}]}
        )
    )
    configure_vertex(monkeypatch, module, session)
    llm = module.LlmVertex()

    assert llm.basic_prompt("Return JSON", response_format="json") == '{"ok":true}'
    assert session.posts[0][1]["json"]["generationConfig"]["responseMimeType"] == "application/json"


def test_json_prompt_retries_malformed_response(monkeypatch):
    module = load_vertex_module()
    session = FakeSession(
        FakeResponse(
            {"candidates": [{"content": {"parts": [{"text": "{malformed"}]}}]}
        ),
        FakeResponse(
            {"candidates": [{"content": {"parts": [{"text": '{"ok":true}'}]}}]}
        ),
    )
    configure_vertex(monkeypatch, module, session)
    llm = module.LlmVertex()

    assert llm.basic_prompt("Return JSON", response_format="json") == '{"ok":true}'
    assert len(session.posts) == 2


def test_basic_prompt_returns_none_on_vertex_error(monkeypatch):
    module = load_vertex_module()
    session = FakeSession(FakeResponse(error=RuntimeError("unavailable")))
    configure_vertex(monkeypatch, module, session)
    llm = module.LlmVertex()

    assert llm.basic_prompt("Tag this conversation") is None


def test_embeddings_use_vertex_with_compatible_dimensions(monkeypatch):
    module = load_vertex_module()
    session = FakeSession(
        FakeResponse(
            {"predictions": [{"embeddings": {"values": [0.25, -0.5]}}]}
        )
    )
    configure_vertex(monkeypatch, module, session)
    llm = module.LlmVertex()

    assert llm.get_vector_embeddings("line\nbreak") == [0.25, -0.5]
    assert session.posts == [
        (
            "https://aiplatform.googleapis.com/v1/projects/test-project/locations/global/"
            "publishers/google/models/gemini-embedding-001:predict",
            {
                "json": {
                    "instances": [
                        {"content": "line break", "task_type": "SEMANTIC_SIMILARITY"}
                    ],
                    "parameters": {"outputDimensionality": 1536},
                },
                "timeout": 24,
            },
        )
    ]


def test_missing_project_is_rejected(monkeypatch):
    module = load_vertex_module()
    session = FakeSession()
    configure_vertex(
        monkeypatch,
        module,
        session,
        overrides={("env", "GOOGLE_CLOUD_PROJECT"): None},
    )
    monkeypatch.setattr(
        module,
        "google_auth_default",
        lambda **kwargs: (object(), None),
    )

    with pytest.raises(ValueError, match="GOOGLE_CLOUD_PROJECT"):
        module.LlmVertex()


def test_factory_selects_vertex_backend(monkeypatch):
    module = load_vertex_module()
    session = FakeSession()
    configure_vertex(monkeypatch, module, session)

    from conversationgenome.llm import llm_factory

    monkeypatch.setattr(
        llm_factory.c,
        "get",
        lambda section, key, default=None: (
            False if (section, key) == ("system", "llm_overrides_locked") else default
        ),
    )

    llm = llm_factory.get_llm_backend(llm_type_override="vertex")

    assert isinstance(llm, module.LlmVertex)
