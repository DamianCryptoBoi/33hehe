from unittest.mock import Mock, patch

from conversationgenome.llm.prompt_manager import prompt_manager
from conversationgenome.llm.llm_factory import get_llm_backend


def test_tagging_prompts_request_scoring_friendly_output():
    prompts = [
        prompt_manager.conversation_to_metadata_prompt("<conversation />"),
        prompt_manager.conversation_to_metadata_coding_prompt("<conversation />"),
        prompt_manager.website_to_metadata_prompt("webpage"),
        prompt_manager.website_to_metadata_coding_prompt("webpage"),
    ]

    for prompt in prompts:
        normalized = prompt.lower()
        assert "18-20" in normalized
        assert "comma-delimited" in normalized
        assert "no json" in normalized
        assert "semantic" in normalized
        assert "abbreviation" in normalized


def test_combine_metadata_tags_uses_topic_combination_prompt(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    llm = get_llm_backend()
    llm.basic_prompt = Mock(return_value="artificial intelligence, machine learning")

    with patch.object(
        prompt_manager,
        "combine_metadata_tags_prompt",
        return_value="topic-combination-prompt",
    ) as build_prompt:
        result = llm.combine_metadata_tags(
            [["artificial intelligence"], ["machine learning"]],
            generateEmbeddings=False,
        )

    build_prompt.assert_called_once_with(
        [["artificial intelligence"], ["machine learning"]]
    )
    llm.basic_prompt.assert_called_once_with("topic-combination-prompt")
    assert result.tags == ["artificial intelligence", "machine learning"]
