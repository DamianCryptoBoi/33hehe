import asyncio
import time
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple

import bittensor as bt
from pydantic import BaseModel

from conversationgenome.api.models.conversation import Conversation
from conversationgenome.llm.llm_factory import get_llm_backend
from conversationgenome.task.Task import Task


def _balanced_fallback(tag_sets: list[list[str]], limit: int = 16) -> list[str]:
    if not tag_sets:
        return []

    output = list(dict.fromkeys(tag_sets[0]))[:10]
    seen = set(output)
    enrichment_sets = tag_sets[1:]
    enrichment_idx = 0

    while len(output) < limit and any(
        enrichment_idx < len(tags) for tags in enrichment_sets
    ):
        for tags in enrichment_sets:
            if enrichment_idx < len(tags) and tags[enrichment_idx] not in seen:
                output.append(tags[enrichment_idx])
                seen.add(tags[enrichment_idx])
                if len(output) == limit:
                    return output
        enrichment_idx += 1

    for tag in tag_sets[0][10:]:
        if tag not in seen:
            output.append(tag)
            seen.add(tag)
            if len(output) == limit:
                break
    return output


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
        started_at = time.monotonic()
        request_timeout = max(self.timeout - 1, 0.01)
        llml = get_llm_backend(task_type=self.type, request_timeout=request_timeout)

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

            fallback_tags = _balanced_fallback(all_tags)
            response_margin = min(1.5, self.timeout / 2)
            remaining = self.timeout - response_margin - (time.monotonic() - started_at)
            if remaining > 0:
                try:
                    combined_result = await asyncio.wait_for(
                        asyncio.to_thread(
                            llml.combine_metadata_tags,
                            all_tags,
                            generateEmbeddings=False,
                        ),
                        timeout=remaining,
                    )
                    if combined_result and combined_result.tags:
                        return {
                            "tags": combined_result.tags[:20],
                            "vectors": combined_result.vectors,
                        }
                except TimeoutError:
                    bt.logging.warning("Metadata combination timed out; using fallback tags")
                except Exception as error:
                    bt.logging.warning(
                        f"Metadata combination failed; using fallback tags: {error}"
                    )

            output = {"tags": fallback_tags, "vectors": None}
        except Exception as e:
            bt.logging.error(f"Error during mining: {e}")
            raise e

        return output
