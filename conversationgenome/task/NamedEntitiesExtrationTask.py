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


class NamedEntitiesExtractionTaskInputData(BaseModel):
    window_idx: int = -1
    window: Optional[List[Tuple[int, str]]] = None
    participants: Optional[List[str]] = None


class NamedEntitiesExtractionTaskInput(BaseModel):
    guid: str
    input_type: Literal["document"]
    data: NamedEntitiesExtractionTaskInputData


class NamedEntitiesExtractionTask(Task):
    type: Literal["named_entities_extraction"] = "named_entities_extraction"
    input: Optional[NamedEntitiesExtractionTaskInput] = None

    async def mine(self) -> dict[str, list]:
        llml = get_llm_backend()

        if not len(self.input.data.window):
            bt.logging.warning('Received empty window in miner, returning no tags')
            return {"tags": []}
        
        try:
            _, main_content = self.input.data.window[0]
            calls = [
                asyncio.to_thread(
                    llml.raw_transcript_to_named_entities,
                    main_content,
                    generateEmbeddings=False,
                )
            ]
            calls.extend(
                asyncio.to_thread(
                    llml.enrichment_to_NER,
                    content,
                    generateEmbeddings=False,
                )
                for _, content in self.input.data.window[1:]
            )
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
