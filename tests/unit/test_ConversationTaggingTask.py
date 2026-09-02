from unittest.mock import MagicMock, Mock
from unittest.mock import patch

import pytest

from tests.mocks.DummyData import DummyData

@pytest.mark.asyncio
async def test_mine_returns_expected_tags_and_vectors():
    mock_llml = MagicMock()
    mock_result = Mock()
    mock_result.tags = ["greeting"]
    mock_result.vectors = [[0.1, 0.2]]
    mock_llml.conversation_to_metadata = Mock(return_value=mock_result)
    # Mock LlmLib and its conversation_to_metadata method
    with patch("conversationgenome.task.ConversationTaggingTask.get_llm_backend", return_value=mock_llml):
        task = DummyData.conversation_tagging_task()
        task.prompt_chain = [type("Prompt", (), {"prompt_template": "Tag the conversation."})()]

        result = await task.mine()

        assert result["tags"] == ["greeting"]
        assert result["vectors"] == [[0.1, 0.2]]
        call_kwargs = mock_llml.conversation_to_metadata.call_args.kwargs
        assert call_kwargs["generateEmbeddings"] is False
        assert getattr(call_kwargs["conversation"], "miner_task_prompt") == "Tag the conversation."


@pytest.mark.asyncio
async def test_mine_handles_empty_tags_and_vectors():
    mock_llml = MagicMock()
    mock_result = Mock()
    mock_result.tags = []
    mock_result.vectors = []
    mock_llml.conversation_to_metadata = Mock(return_value=mock_result)
    # Mock LlmLib and its conversation_to_metadata method to return empty tags and vectors
    with patch("conversationgenome.task.ConversationTaggingTask.get_llm_backend", return_value=mock_llml):
        task = DummyData.conversation_tagging_task()
        task.prompt_chain = [type("Prompt", (), {"prompt_template": "Tag the conversation."})()]

        result = await task.mine()

        assert result["tags"] == []
        assert result["vectors"] == []


@pytest.mark.asyncio
async def test_mine_handles_none_tags_and_vectors():
    # Mock LlmLib and its conversation_to_metadata method to return None for tags and vectors
    mock_llml = MagicMock()
    mock_result = Mock()
    mock_result.tags = None
    mock_result.vectors = None
    mock_llml.conversation_to_metadata = Mock(return_value=mock_result)
    with patch("conversationgenome.task.ConversationTaggingTask.get_llm_backend", return_value=mock_llml):
        task = DummyData.conversation_tagging_task()
        task.prompt_chain = [type("Prompt", (), {"prompt_template": "Tag the conversation."})()]

        result = await task.mine()

        assert result["tags"] is None
        assert result["vectors"] is None


@pytest.mark.asyncio
async def test_mine_handles_exception_from_llmlib_raises_error():
    # Mock LlmLib to raise an exception
    mock_llml = MagicMock()
    mock_llml.conversation_to_metadata = Mock(side_effect=Exception("Mining failed"))
    with patch("conversationgenome.task.ConversationTaggingTask.get_llm_backend", return_value=mock_llml):

        task = DummyData.conversation_tagging_task()
        task.prompt_chain = [type("Prompt", (), {"prompt_template": "Tag the conversation."})()]

        with pytest.raises(Exception, match="Mining failed"):
            await task.mine()


@pytest.mark.asyncio
async def test_mine_uses_validator_aligned_enrichment_and_named_entity_combiner():
    mock_llml = MagicMock()
    primary_result = Mock(tags=["east german military"], vectors=None)
    enrichment_result = Mock(tags=["united states congress"], vectors=None)
    combined_result = Mock(
        tags=["east german military", "united states congress"],
        vectors=None,
    )
    mock_llml.conversation_to_metadata.return_value = primary_result
    mock_llml.enrichment_to_metadata.return_value = enrichment_result
    mock_llml.combine_named_entities.return_value = combined_result

    with patch("conversationgenome.task.ConversationTaggingTask.get_llm_backend", return_value=mock_llml):
        task = DummyData.conversation_tagging_task()
        task.input.data.enrichment_lines = [(0, "United States Congress and Senate")]

        result = await task.mine()

    mock_llml.enrichment_to_metadata.assert_called_once_with(
        "United States Congress and Senate",
        generateEmbeddings=False,
        input_categories=task.input.input_categories,
        validator_aligned=True,
    )
    mock_llml.combine_named_entities.assert_called_once_with(
        [["east german military"], ["united states congress"]],
        generateEmbeddings=False,
    )
    mock_llml.combine_metadata_tags.assert_not_called()
    assert result == {
        "tags": ["east german military", "united states congress"],
        "vectors": None,
    }


@pytest.mark.asyncio
async def test_mine_preserves_all_extracted_tags_when_combination_fails():
    mock_llml = MagicMock()
    mock_llml.conversation_to_metadata.return_value = Mock(
        tags=["east german military", "bernd schwipper"], vectors=None
    )
    mock_llml.enrichment_to_metadata.return_value = Mock(
        tags=["united states congress", "bernd schwipper"], vectors=None
    )
    mock_llml.combine_named_entities.return_value = None

    with patch("conversationgenome.task.ConversationTaggingTask.get_llm_backend", return_value=mock_llml):
        task = DummyData.conversation_tagging_task()
        task.input.data.enrichment_lines = [(0, "United States Congress")]
        result = await task.mine()

    assert result == {
        "tags": ["east german military", "bernd schwipper", "united states congress"],
        "vectors": None,
    }
