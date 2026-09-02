from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple

import bittensor as bt
from pydantic import BaseModel

from conversationgenome.llm.llm_factory import get_llm_backend
from conversationgenome.task.Task import Task


class SkillTaskInputData(BaseModel):
    window: List[Tuple[int, str]]
    participants: Optional[List[str]] = None

class SkillTaskInput(BaseModel):
    guid: str
    input_type: Literal["skill"]
    data: SkillTaskInputData
    input_categories: Optional[List[str]] = None


class SkillGenerationTask(Task):
    type: Literal["skill_generation"] = "skill_generation"
    input: Optional[SkillTaskInput] = None

    async def mine(self) -> dict[str, list]:
        llml = get_llm_backend()

        try:
            all_tags = []

            # The window contains the skill markdown line(s) written by the bundle's setup
            for idx, (line_idx, content) in enumerate(self.input.data.window):
                result = llml.skill_to_metadata(content, generateEmbeddings=False, input_categories=self.input.input_categories)

                if result and result.tags:
                    all_tags.append(result.tags)

            if not all_tags:
                output = {"tags": [], "vectors": None}
            else:
                fallback_tags = list(dict.fromkeys(tag for tags in all_tags for tag in tags))[:20]
                if len(all_tags) == 1:
                    return {"tags": fallback_tags, "vectors": None}

                combined_result = llml.combine_metadata_tags(all_tags, generateEmbeddings=False)
                output = (
                    {"tags": combined_result.tags, "vectors": combined_result.vectors}
                    if combined_result
                    else {"tags": fallback_tags, "vectors": None}
                )

        except Exception as e:
            bt.logging.error(f"Error during mining: {e}")
            raise e

        return output
