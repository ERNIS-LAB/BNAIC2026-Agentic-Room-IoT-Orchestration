from langchain_core.outputs import ChatGenerationChunk
from langchain_openai import ChatOpenAI


class ChatOpenAIWithTimings(ChatOpenAI):
    """ChatOpenAI subclass that preserves the llama.cpp `timings` field on the AIMessage
    and captures `reasoning_content` from streaming chunks (thinking mode).
    """

    stream_usage: bool = True

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        # llama.cpp (and many OpenAI-compatible backends) reject messages
        # with content=None, which LangChain emits for assistant messages
        # that carry tool_calls but no text. Rehydrate to empty string.
        for msg in payload.get("messages", []):
            if msg.get("content") is None:
                msg["content"] = ""
        return payload

    def _create_chat_result(self, response, generation_info=None):
        result = super()._create_chat_result(response, generation_info)
        response_dict = (
            response if isinstance(response, dict) else response.model_dump()
        )
        timings = response_dict.get("timings")
        if timings:
            for generation in result.generations:
                generation.message.additional_kwargs["timings"] = timings
        return result

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: dict | None,
    ) -> ChatGenerationChunk | None:
        result = super()._convert_chunk_to_generation_chunk(
            chunk,
            default_chunk_class,
            base_generation_info,
        )
        if result is not None:
            # Extract reasoning_content from the delta (DeepSeek-format thinking).
            # llama.cpp with --reasoning-format deepseek sends thinking text in
            # delta.reasoning_content, separate from delta.content.
            choices = chunk.get("choices", []) or chunk.get("chunk", {}).get(
                "choices", []
            )
            if choices:
                delta = choices[0].get("delta") or {}
                reasoning = delta.get("reasoning_content")
                if reasoning:
                    result.message.additional_kwargs["reasoning_content"] = reasoning
        return result
