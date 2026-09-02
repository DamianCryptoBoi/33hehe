import asyncio
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
        llml = get_llm_backend(request_timeout=self.timeout - 2)

        try:
            calls = [
                asyncio.to_thread(
                    llml.skill_to_metadata,
                    content,
                    generateEmbeddings=False,
                    input_categories=self.input.input_categories,
                )
                for _, content in self.input.data.window
            ]
            results = await asyncio.gather(*calls)
            all_tags = [result.tags for result in results if result and result.tags]

            if not all_tags:
                output = {"tags": [], "vectors": None}
            else:
                output = {
                    "tags": list(dict.fromkeys(tag for tags in all_tags for tag in tags))[:20],
                    "vectors": None,
                }

        except Exception as e:
            bt.logging.error(f"Error during mining: {e}")
            raise e

        return output
