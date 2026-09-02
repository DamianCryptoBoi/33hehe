import threading
import time
from unittest.mock import MagicMock, Mock
from unittest.mock import patch

import pytest

from conversationgenome.prompt_chain.PromptChainStep import PromptChainStep
from conversationgenome.task.SkillGenerationTask import SkillGenerationTask, SkillTaskInput, SkillTaskInputData


def _make_task(window):
    return SkillGenerationTask(
        mode="local",
        api_version=1.4,
        guid="test-guid",
        bundle_guid="bundle-guid",
        type="skill_generation",
        input=SkillTaskInput(
            guid="input-guid",
            input_type="skill",
            data=SkillTaskInputData(
                window=window,
                participants=[]
            )
        ),
        prompt_chain=[PromptChainStep(
            step=0,
            id="skill_001",
            crc=12345,
            title="Infer tags",
            name="infer_tags_for_skill",
            description="Infer descriptive tags for a skill document",
            type="inference",
            input_path="skill",
            prompt_template="Infer tags for the skill",
            output_variable="final_output",
            output_type="List[str]"
        )]
    )


@pytest.mark.asyncio
async def test_mine_returns_tags_and_vectors():
    task = _make_task([(0, "# Parse .docx\n\nInstructions to parse docx files.")])

    mock_llml = MagicMock()
    skill_result = Mock()
    skill_result.tags = ["docx", "parsing"]
    skill_result.vectors = None
    mock_llml.skill_to_metadata = Mock(return_value=skill_result)

    with patch("conversationgenome.task.SkillGenerationTask.get_llm_backend", return_value=mock_llml) as mock_get_llm:
        result = await task.mine()

    mock_get_llm.assert_called_once_with(request_timeout=10)
    assert result["tags"] == ["docx", "parsing"]
    assert result["vectors"] is None
    mock_llml.skill_to_metadata.assert_called_once_with(
        "# Parse .docx\n\nInstructions to parse docx files.", generateEmbeddings=False, input_categories=None
    )
    mock_llml.combine_metadata_tags.assert_not_called()


@pytest.mark.asyncio
async def test_mine_returns_locally_merged_extracted_tags():
    task = _make_task([(0, "first"), (1, "second")])
    mock_llml = MagicMock()
    mock_llml.skill_to_metadata.side_effect = lambda content, **_: (
        Mock(tags=["authentication", "magic links"])
        if content == "first"
        else Mock(tags=["email security", "authentication"])
    )

    with patch("conversationgenome.task.SkillGenerationTask.get_llm_backend", return_value=mock_llml):
        result = await task.mine()

    assert result == {
        "tags": ["authentication", "magic links", "email security"],
        "vectors": None,
    }
    mock_llml.combine_metadata_tags.assert_not_called()


@pytest.mark.asyncio
async def test_mine_caps_local_tags_at_twenty_in_original_order():
    task = _make_task([(0, "first"), (1, "second")])
    mock_llml = MagicMock()
    mock_llml.skill_to_metadata.side_effect = lambda content, **_: (
        Mock(tags=[f"tag-{index}" for index in range(15)])
        if content == "first"
        else Mock(tags=["tag-0"] + [f"tag-{index}" for index in range(15, 25)])
    )

    with patch("conversationgenome.task.SkillGenerationTask.get_llm_backend", return_value=mock_llml):
        result = await task.mine()

    assert result == {
        "tags": [f"tag-{index}" for index in range(20)],
        "vectors": None,
    }
    mock_llml.combine_metadata_tags.assert_not_called()


@pytest.mark.asyncio
async def test_mine_handles_empty_window():
    task = _make_task([])
    mock_llml = MagicMock()

    with patch("conversationgenome.task.SkillGenerationTask.get_llm_backend", return_value=mock_llml):
        result = await task.mine()

    assert result["tags"] == []
    assert result["vectors"] is None
    mock_llml.skill_to_metadata.assert_not_called()
    mock_llml.combine_metadata_tags.assert_not_called()


@pytest.mark.asyncio
async def test_mine_handles_none_result():
    task = _make_task([(0, "Test skill content")])
    mock_llml = MagicMock()
    mock_llml.skill_to_metadata = Mock(return_value=None)
    with patch("conversationgenome.task.SkillGenerationTask.get_llm_backend", return_value=mock_llml):
        result = await task.mine()

    assert result["tags"] == []
    assert result["vectors"] is None


@pytest.mark.asyncio
async def test_mine_raises_on_llm_exception():
    task = _make_task([(0, "Test skill content")])
    mock_llml = MagicMock()
    mock_llml.skill_to_metadata = Mock(side_effect=Exception("LLM Error"))

    with patch("conversationgenome.task.SkillGenerationTask.get_llm_backend", return_value=mock_llml):
        with pytest.raises(Exception, match="LLM Error"):
            await task.mine()


@pytest.mark.asyncio
async def test_mine_runs_skill_line_extractions_concurrently():
    task = _make_task([(0, "first"), (1, "second")])
    mock_llml = MagicMock()
    lock = threading.Lock()
    active = 0
    max_active = 0

    def observe(content):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return Mock(tags=[content], vectors=None)

    mock_llml.skill_to_metadata.side_effect = lambda content, **_: observe(content)

    with patch("conversationgenome.task.SkillGenerationTask.get_llm_backend", return_value=mock_llml):
        await task.mine()

    assert max_active == 2
