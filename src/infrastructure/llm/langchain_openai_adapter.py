from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from domain.ports.llm_port import LLMPort

_MESSAGE_TYPES = {
    "system": SystemMessage,
    "user": HumanMessage,
    "assistant": AIMessage,
}


def _text_content(response: Any) -> str:
    """Extract the assistant's text content from an ``AIMessage``.

    ``AIMessage.content`` can be a plain string or a content-block list for
    multimodal models; the text adapter only produces/consumes text prompts,
    so a non-string payload is rendered deterministically instead of falling
    through an untyped union.
    """
    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content
    return "".join(
        block["text"]
        for block in content
        if isinstance(block, dict) and isinstance(block.get("text"), str)
    )


class LangchainOpenAIAdapter(LLMPort):
    """OpenAI-compatible LLM adapter using langchain-openai's ChatOpenAI.

    Supports both single-turn (``generate``) and multi-turn (``generate_chat``)
    interactions. Works with any OpenAI-compatible endpoint (OpenRouter, etc.).
    """

    def __init__(self, api_key: str, base_url: str, model: str, temperature: float):
        self._llm = ChatOpenAI(
            api_key=SecretStr(api_key),
            base_url=base_url,
            model=model,
            temperature=temperature,
        )

    async def generate(self, system_prompt: str, user_message: str) -> str:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]
        response = await self._llm.ainvoke(messages)
        return _text_content(response)

    async def generate_chat(self, messages: list[dict[str, str]]) -> str:
        lc_messages = [
            _MESSAGE_TYPES[msg["role"]](content=msg["content"])
            for msg in messages
            if msg["role"] in _MESSAGE_TYPES
        ]
        response = await self._llm.ainvoke(lc_messages)
        return _text_content(response)
