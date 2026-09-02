from unittest.mock import MagicMock
from unittest.mock import patch

from conversationgenome.llm.llm_openai import DEFAULT_MODEL
from conversationgenome.llm.llm_openai import LlmOpenAI


def _c_get(overrides):
    def get(section, key, default=None):
        return overrides.get((section, key), default)
    return get


@patch("conversationgenome.llm.llm_openai.OpenAI")
@patch("conversationgenome.llm.llm_openai.c")
def test_model_override_honored_by_default(mock_c, mock_openai):
    mock_c.get.side_effect = _c_get({
        ("env", "OPENAI_API_KEY"): "test-key",
        ("env", "OPENAI_MODEL"): "gpt-custom",
    })

    llm = LlmOpenAI()

    assert llm.model == "gpt-custom"


@patch("conversationgenome.llm.llm_openai.OpenAI")
@patch("conversationgenome.llm.llm_openai.c")
def test_model_override_ignored_when_locked(mock_c, mock_openai):
    mock_c.get.side_effect = _c_get({
        ("env", "OPENAI_API_KEY"): "test-key",
        ("env", "OPENAI_MODEL"): "gpt-custom",
    })

    llm = LlmOpenAI(ignore_model_override=True)

    assert llm.model == DEFAULT_MODEL


@patch("conversationgenome.llm.llm_openai.OpenAI")
@patch("conversationgenome.llm.llm_openai.c")
def test_default_model_used_when_no_override_present(mock_c, mock_openai):
    mock_c.get.side_effect = _c_get({("env", "OPENAI_API_KEY"): "test-key"})

    llm = LlmOpenAI()

    assert llm.model == DEFAULT_MODEL


@patch("conversationgenome.llm.llm_openai.OpenAI")
@patch("conversationgenome.llm.llm_openai.c")
def test_custom_base_url_is_passed_to_openai_client(mock_c, mock_openai):
    mock_c.get.side_effect = _c_get({
        ("env", "OPENAI_API_KEY"): "test-key",
        ("env", "OPENAI_BASE_URL"): "https://api.xah.io/v1",
        ("env", "OPENAI_MODEL"): "deepseek-v4-flash",
    })

    LlmOpenAI()

    mock_openai.assert_called_once_with(
        api_key="test-key",
        base_url="https://api.xah.io/v1",
        timeout=10.0,
        max_retries=0,
    )


@patch("conversationgenome.llm.llm_openai.OpenAI")
@patch("conversationgenome.llm.llm_openai.c")
def test_custom_base_url_uses_configured_model_for_skill_requests(mock_c, mock_openai):
    mock_c.get.side_effect = _c_get({
        ("env", "OPENAI_API_KEY"): "test-key",
        ("env", "OPENAI_BASE_URL"): "https://api.xah.io/v1",
        ("env", "OPENAI_MODEL"): "deepseek-v4-flash",
    })
    llm = LlmOpenAI()
    observed = {}

    def record_prompt(prompt, response_format="text"):
        observed.update(
            model=llm.model,
            reasoning_effort=llm.reasoning_effort,
            service_tier=llm.service_tier,
        )
        return '{"sections":[{"section_id":"s1","title":"One","description":"Test"}]}'

    llm.basic_prompt = record_prompt

    result = llm.skill_request_to_section_map("Build one thing")

    assert result is not None
    assert observed == {
        "model": "deepseek-v4-flash",
        "reasoning_effort": None,
        "service_tier": None,
    }


@patch("conversationgenome.llm.llm_openai.OpenAI")
@patch("conversationgenome.llm.llm_openai.c")
def test_text_prompt_omits_response_format_for_compatible_endpoints(mock_c, mock_openai):
    mock_c.get.side_effect = _c_get({
        ("env", "OPENAI_API_KEY"): "test-key",
        ("env", "OPENAI_BASE_URL"): "https://api.xah.io/v1",
        ("env", "OPENAI_MODEL"): "deepseek-v4-flash",
    })
    completion = MagicMock()
    completion.choices[0].message.content = "done"
    mock_openai.return_value.chat.completions.create.return_value = completion

    llm = LlmOpenAI()
    result = llm.basic_prompt("Hello")

    assert result == "done"
    mock_openai.return_value.chat.completions.create.assert_called_once_with(
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": "Hello"}],
    )


@patch("conversationgenome.llm.llm_openai.OpenAI")
@patch("conversationgenome.llm.llm_openai.c")
def test_fast_transient_completion_error_is_retried_once(mock_c, mock_openai, capsys):
    mock_c.get.side_effect = _c_get({
        ("env", "OPENAI_API_KEY"): "test-key",
        ("env", "OPENAI_MODEL"): "deepseek-v4-flash",
    })
    transient_error = RuntimeError("temporary provider failure")
    transient_error.status_code = 503
    completion = MagicMock()
    completion.choices[0].message.content = "recovered"
    create = mock_openai.return_value.chat.completions.create
    create.side_effect = [transient_error, completion]

    result = LlmOpenAI().basic_prompt("Hello")

    assert result == "recovered"
    assert create.call_count == 2
    assert "RuntimeError: temporary provider failure" in capsys.readouterr().out


@patch("conversationgenome.llm.llm_openai.OpenAI")
@patch("conversationgenome.llm.llm_openai.c")
def test_fast_empty_completion_is_retried_once(mock_c, mock_openai, capsys):
    mock_c.get.side_effect = _c_get({
        ("env", "OPENAI_API_KEY"): "test-key",
    })
    empty_completion = MagicMock()
    empty_completion.choices[0].message.content = ""
    recovered_completion = MagicMock()
    recovered_completion.choices[0].message.content = "recovered"
    create = mock_openai.return_value.chat.completions.create
    create.side_effect = [empty_completion, recovered_completion]

    result = LlmOpenAI().basic_prompt("Hello")

    assert result == "recovered"
    assert create.call_count == 2
    assert "empty response content" in capsys.readouterr().out


@patch("conversationgenome.llm.llm_openai.OpenAI")
@patch("conversationgenome.llm.llm_openai.c")
def test_non_transient_completion_error_is_logged_without_retry(mock_c, mock_openai, capsys):
    mock_c.get.side_effect = _c_get({
        ("env", "OPENAI_API_KEY"): "test-key",
    })
    create = mock_openai.return_value.chat.completions.create
    create.side_effect = ValueError("invalid provider response")

    result = LlmOpenAI().basic_prompt("Hello")

    assert result is None
    create.assert_called_once()
    assert "ValueError: invalid provider response" in capsys.readouterr().out


@patch("conversationgenome.llm.llm_openai.time.monotonic", side_effect=[0.0, 2.0])
@patch("conversationgenome.llm.llm_openai.OpenAI")
@patch("conversationgenome.llm.llm_openai.c")
def test_slow_transient_completion_error_is_not_retried(mock_c, mock_openai, mock_monotonic):
    mock_c.get.side_effect = _c_get({
        ("env", "OPENAI_API_KEY"): "test-key",
    })
    transient_error = RuntimeError("slow provider failure")
    transient_error.status_code = 503
    create = mock_openai.return_value.chat.completions.create
    create.side_effect = transient_error

    result = LlmOpenAI().basic_prompt("Hello")

    assert result is None
    create.assert_called_once()
    assert mock_monotonic.call_count == 2
