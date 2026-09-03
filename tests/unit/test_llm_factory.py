import json
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from conversationgenome.llm.llm_factory import _present_llm_override_vars
from conversationgenome.llm.llm_factory import configure_llm_override_lockdown
from conversationgenome.llm.llm_factory import get_llm_backend
from conversationgenome.llm.llm_openai import LlmOpenAI


MINER_LLM_CONFIG = {
    "default": {
        "provider": "vertex",
        "model": "gemini-3.8-flash",
        "reasoning_effort": "low",
    },
    "tasks": {
        "conversation_tagging": {
            "provider": "openai",
            "base_url": "https://api.xah.io/v1",
            "model": "gpt-5.4",
            "reasoning_effort": "none",
        }
    },
}


def _write_miner_llm_config(tmp_path, config=MINER_LLM_CONFIG):
    config_path = tmp_path / "miner_llm_config.json"
    config_path.write_text(json.dumps(config))
    return config_path


def _c_get(overrides):
    def get(section, key, default=None):
        return overrides.get((section, key), default)
    return get


@patch("conversationgenome.llm.llm_factory.c")
def test_present_llm_override_vars_none_set(mock_c):
    mock_c.get.side_effect = _c_get({})
    assert _present_llm_override_vars() == []


@patch("conversationgenome.llm.llm_factory.c")
def test_present_llm_override_vars_each_detected(mock_c):
    for var in [
        "LLM_TYPE_OVERRIDE",
        "OPENAI_MODEL",
        "OPENAI_BASE_URL",
        "OPENAI_EMBEDDINGS_MODEL_OVERRIDE",
    ]:
        mock_c.get.side_effect = _c_get({("env", var): "some-value"})
        assert _present_llm_override_vars() == [var]


@patch("conversationgenome.llm.llm_factory.bt")
@patch("conversationgenome.llm.llm_factory.c")
def test_configure_lockdown_mainnet_no_override(mock_c, mock_bt):
    mock_c.get.side_effect = _c_get({("network", "mainnet"): 33})
    mock_c.set = MagicMock()

    result = configure_llm_override_lockdown(33)

    assert result is True
    mock_c.set.assert_called_once_with("system", "llm_overrides_locked", True)
    mock_bt.logging.warning.assert_not_called()


@patch("conversationgenome.llm.llm_factory.bt")
@patch("conversationgenome.llm.llm_factory.c")
def test_configure_lockdown_mainnet_with_override_warns_and_locks(mock_c, mock_bt):
    mock_c.get.side_effect = _c_get({
        ("network", "mainnet"): 33,
        ("env", "LLM_TYPE_OVERRIDE"): "anthropic",
    })
    mock_c.set = MagicMock()

    result = configure_llm_override_lockdown(33)

    assert result is True
    mock_c.set.assert_called_once_with("system", "llm_overrides_locked", True)
    mock_bt.logging.warning.assert_called_once()
    assert "ignored" in mock_bt.logging.warning.call_args[0][0]


@patch("conversationgenome.llm.llm_factory.bt")
@patch("conversationgenome.llm.llm_factory.c")
def test_configure_lockdown_testnet_with_override_warns_but_allows(mock_c, mock_bt):
    mock_c.get.side_effect = _c_get({
        ("network", "mainnet"): 33,
        ("env", "OPENAI_MODEL"): "gpt-custom",
    })
    mock_c.set = MagicMock()

    result = configure_llm_override_lockdown(138)

    assert result is False
    mock_c.set.assert_called_once_with("system", "llm_overrides_locked", False)
    mock_bt.logging.warning.assert_called_once()
    assert "Honoring" in mock_bt.logging.warning.call_args[0][0]


@patch("conversationgenome.llm.llm_factory.bt")
@patch("conversationgenome.llm.llm_factory.c")
def test_configure_lockdown_testnet_no_override_no_warning(mock_c, mock_bt):
    mock_c.get.side_effect = _c_get({("network", "mainnet"): 33})
    mock_c.set = MagicMock()

    result = configure_llm_override_lockdown(138)

    assert result is False
    mock_bt.logging.warning.assert_not_called()


@patch("conversationgenome.llm.llm_openai.OpenAI")
@patch("conversationgenome.llm.llm_openai.c")
@patch("conversationgenome.llm.llm_factory.c")
def test_get_llm_backend_locked_forces_openai_ignoring_explicit_override(mock_factory_c, mock_openai_c, mock_openai_client):
    mock_factory_c.get.side_effect = _c_get({("system", "llm_overrides_locked"): True})
    mock_openai_c.get.side_effect = _c_get({
        ("env", "OPENAI_API_KEY"): "test-key",
        ("env", "OPENAI_BASE_URL"): "https://api.xah.io/v1",
    })

    llm = get_llm_backend(llm_type_override="anthropic")

    assert isinstance(llm, LlmOpenAI)
    assert llm.model == "gpt-5.2"
    mock_openai_client.assert_called_once_with(
        api_key="test-key",
        timeout=10.0,
        max_retries=0,
    )


@patch("conversationgenome.llm.llm_openai.OpenAI")
@patch("conversationgenome.llm.llm_openai.c")
@patch("conversationgenome.llm.llm_factory.c")
def test_get_llm_backend_unlocked_honors_override(mock_factory_c, mock_openai_c, mock_openai_client):
    mock_factory_c.get.side_effect = _c_get({
        ("system", "llm_overrides_locked"): False,
        ("env", "LLM_TYPE_OVERRIDE"): None,
    })

    llm = get_llm_backend(llm_type_override=None)

    assert isinstance(llm, LlmOpenAI)


@patch("conversationgenome.llm.llm_openai.OpenAI")
@patch("conversationgenome.llm.llm_openai.c")
@patch("conversationgenome.llm.llm_factory.c")
def test_get_llm_backend_passes_request_timeout_to_openai(mock_factory_c, mock_openai_c, mock_openai_client):
    mock_factory_c.get.side_effect = _c_get({
        ("system", "llm_overrides_locked"): False,
        ("env", "LLM_TYPE_OVERRIDE"): None,
    })
    mock_openai_c.get.side_effect = _c_get({("env", "OPENAI_API_KEY"): "test-key"})

    get_llm_backend(request_timeout=22)

    mock_openai_client.assert_called_once_with(
        api_key="test-key",
        timeout=22,
        max_retries=0,
    )


@patch("conversationgenome.llm.llm_factory.c")
def test_get_llm_backend_never_locked_for_miner_process(mock_c):
    """Miners never call configure_llm_override_lockdown, so the flag defaults to False
    and get_llm_backend() honors overrides exactly as before."""
    with patch("conversationgenome.llm.llm_openrouter.c") as mock_openrouter_c, \
         patch("conversationgenome.llm.llm_openrouter.OpenAI"):
        mock_c.get.side_effect = _c_get({
            ("system", "llm_overrides_locked"): False,
            ("env", "LLM_TYPE_OVERRIDE"): "openrouter",
        })
        mock_openrouter_c.get.side_effect = _c_get({
            ("env", "OPENROUTER_API_KEY"): "test-key",
            ("env", "OPENROUTER_MODEL"): "test-model",
        })

        from conversationgenome.llm.llm_openrouter import LlmOpenRouter

        llm = get_llm_backend()

        assert isinstance(llm, LlmOpenRouter)


@patch("conversationgenome.llm.llm_openai.OpenAI")
def test_conversation_route_uses_configured_openai_endpoint(mock_openai, monkeypatch, tmp_path):
    config_path = _write_miner_llm_config(tmp_path)
    monkeypatch.setenv("MINER_LLM_CONFIG", str(config_path))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    llm = get_llm_backend(task_type="conversation_tagging", request_timeout=10)

    assert isinstance(llm, LlmOpenAI)
    assert llm.model == "gpt-5.4"
    assert llm.reasoning_effort == "none"
    mock_openai.assert_called_once_with(
        api_key="test-key",
        base_url="https://api.xah.io/v1",
        timeout=10,
        max_retries=0,
    )


def test_other_task_route_uses_configured_vertex_model(monkeypatch, tmp_path):
    config_path = _write_miner_llm_config(tmp_path)
    monkeypatch.setenv("MINER_LLM_CONFIG", str(config_path))
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")

    with patch(
        "conversationgenome.llm.llm_vertex.google_auth_default",
        return_value=(object(), "adc-project"),
    ), patch("conversationgenome.llm.llm_vertex.AuthorizedSession"):
        llm = get_llm_backend(task_type="survey_tagging", request_timeout=10)

    assert llm.model == "gemini-3.8-flash"
    assert llm.reasoning_effort == "LOW"


def test_miner_config_rejects_unknown_task(monkeypatch, tmp_path):
    config = {
        **MINER_LLM_CONFIG,
        "tasks": {
            **MINER_LLM_CONFIG["tasks"],
            "unknown_task": {"provider": "openai", "model": "gpt-5.4"},
        },
    }
    config_path = _write_miner_llm_config(tmp_path, config)
    monkeypatch.setenv("MINER_LLM_CONFIG", str(config_path))

    with pytest.raises(ValueError, match="unknown_task"):
        get_llm_backend(task_type="conversation_tagging")


def test_miner_config_rejects_unknown_route_keys(monkeypatch, tmp_path):
    config = {
        **MINER_LLM_CONFIG,
        "default": {**MINER_LLM_CONFIG["default"], "api_key": "secret"},
    }
    config_path = _write_miner_llm_config(tmp_path, config)
    monkeypatch.setenv("MINER_LLM_CONFIG", str(config_path))

    with pytest.raises(ValueError, match="api_key"):
        get_llm_backend(task_type="survey_tagging")


def test_miner_config_rejects_unknown_top_level_keys(monkeypatch, tmp_path):
    config = {**MINER_LLM_CONFIG, "api_key": "secret"}
    config_path = _write_miner_llm_config(tmp_path, config)
    monkeypatch.setenv("MINER_LLM_CONFIG", str(config_path))

    with pytest.raises(ValueError, match="api_key"):
        get_llm_backend(task_type="survey_tagging")


def test_miner_config_rejects_invalid_vertex_reasoning(monkeypatch, tmp_path):
    config = {
        **MINER_LLM_CONFIG,
        "default": {**MINER_LLM_CONFIG["default"], "reasoning_effort": "none"},
    }
    config_path = _write_miner_llm_config(tmp_path, config)
    monkeypatch.setenv("MINER_LLM_CONFIG", str(config_path))

    with pytest.raises(ValueError, match="reasoning_effort.*none"):
        get_llm_backend(task_type="survey_tagging")


def test_get_llm_backend_rejects_unknown_task_type(monkeypatch, tmp_path):
    config_path = _write_miner_llm_config(tmp_path)
    monkeypatch.setenv("MINER_LLM_CONFIG", str(config_path))

    with pytest.raises(ValueError, match="conversation_taggin"):
        get_llm_backend(task_type="conversation_taggin")


def test_startup_check_fails_when_a_required_route_cannot_generate(monkeypatch, tmp_path):
    from conversationgenome.llm import llm_factory

    config_path = _write_miner_llm_config(tmp_path)
    monkeypatch.setenv("MINER_LLM_CONFIG", str(config_path))
    working_backend = MagicMock()
    working_backend.basic_prompt.return_value = "OK"
    failing_backend = MagicMock()
    failing_backend.basic_prompt.return_value = None

    with patch.object(
        llm_factory,
        "_create_llm_backend",
        side_effect=[working_backend, failing_backend],
    ):
        with pytest.raises(RuntimeError, match="conversation_tagging.*openai.*gpt-5.4"):
            llm_factory.check_miner_llm_backends(request_timeout=5)
