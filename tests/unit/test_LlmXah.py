import threading
from types import SimpleNamespace

from conversationgenome.llm.llm_xah import LlmXah


PRIMARY = "deepseek-v4-flash"
FALLBACK = "levuphong2909/gemini-3.5-flash-high"


def _response(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def _llm(create):
    llm = LlmXah.__new__(LlmXah)
    llm.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    llm.primary_model = PRIMARY
    llm.fallback_model = FALLBACK
    return llm


def test_primary_wins_after_both_requests_have_started():
    both_started = threading.Barrier(2)
    release_fallback = threading.Event()
    started_models = []

    def create(**kwargs):
        model = kwargs["model"]
        started_models.append(model)
        both_started.wait(timeout=2)
        if model == PRIMARY:
            return _response("primary result")
        release_fallback.wait(timeout=2)
        return _response("fallback result")

    try:
        result = _llm(create).basic_prompt("prompt")
    finally:
        release_fallback.set()

    assert result == "primary result"
    assert set(started_models) == {PRIMARY, FALLBACK}


def test_primary_error_returns_already_running_fallback():
    both_started = threading.Barrier(2)

    def create(**kwargs):
        both_started.wait(timeout=2)
        if kwargs["model"] == PRIMARY:
            raise RuntimeError("primary unavailable")
        return _response("fallback result")

    assert _llm(create).basic_prompt("prompt") == "fallback result"


def test_invalid_primary_json_returns_valid_fallback_json():
    both_started = threading.Barrier(2)

    def create(**kwargs):
        assert kwargs["response_format"] == {"type": "json_object"}
        both_started.wait(timeout=2)
        if kwargs["model"] == PRIMARY:
            return _response("not json")
        return _response('{"skill": "fallback"}')

    result = _llm(create).basic_prompt("prompt", response_format="json")

    assert result == '{"skill": "fallback"}'


def test_text_requests_do_not_send_null_response_format():
    both_started = threading.Barrier(2)

    def create(**kwargs):
        if "response_format" in kwargs:
            raise RuntimeError("XAH rejects response_format=null")
        both_started.wait(timeout=2)
        return _response(kwargs["model"])

    assert _llm(create).basic_prompt("prompt") == PRIMARY
