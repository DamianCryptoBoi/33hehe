import asyncio
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple

import bittensor as bt
from pydantic import BaseModel

from conversationgenome.api.models.conversation import Conversation
from conversationgenome.llm.llm_factory import get_llm_backend
from conversationgenome.task.Task import Task


class ConversationTaskInputData(BaseModel):
    window_idx: int = -1
    window: Optional[List[Tuple[int, str]]] = None
    participants: Optional[List[str]] = None
    enrichment_lines: Optional[List[Tuple[int, str]]] = None


class ConversationTaskInput(BaseModel):
    guid: str
    input_type: Literal["conversation"]
    data: ConversationTaskInputData
    input_categories: Optional[List[str]] = None


class ConversationTaggingTask(Task):
    type: Literal["conversation_tagging"] = "conversation_tagging"
    input: Optional[ConversationTaskInput] = None

    async def mine(self) -> dict[str, list]:
        llml = get_llm_backend(request_timeout=self.timeout - 2)

        try:
            conversation = Conversation(
                guid=self.input.guid,
                lines=self.input.data.window,
                miner_task_prompt=self.prompt_chain[0].prompt_template,
                input_categories=self.input.input_categories
            )

            enrichment_lines = self.input.data.enrichment_lines or []
            calls = [
                asyncio.to_thread(
                    llml.conversation_to_metadata,
                    conversation=conversation,
                    generateEmbeddings=False,
                )
            ]
            calls.extend(
                asyncio.to_thread(
                    llml.enrichment_to_metadata,
                    content,
                    generateEmbeddings=False,
                    input_categories=self.input.input_categories,
                    validator_aligned=True,
                )
                for _, content in enrichment_lines
            )
            results = await asyncio.gather(*calls)
            result = results[0]

            if not enrichment_lines:
                if result:
                    return {"tags": result.tags, "vectors": result.vectors}
                return {"tags": [], "vectors": None}

            all_tags = [item.tags for item in results if item and item.tags]

            if not all_tags:
                return {"tags": [], "vectors": None}

            output = {
                "tags": list(dict.fromkeys(tag for tags in all_tags for tag in tags))[:20],
                "vectors": None,
            }
        except Exception as e:
            bt.logging.error(f"Error during mining: {e}")
            raise e

        return output
