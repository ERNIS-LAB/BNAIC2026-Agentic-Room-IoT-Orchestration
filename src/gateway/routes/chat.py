import json
import logging
import uuid

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from gateway.config import settings
from gateway.schemas import ChatCompletionRequest, ErrorDetail, ErrorResponse

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_backend_url(model: str) -> str:
    if model != settings.model_name:
        raise HTTPException(
            status_code=404,
            detail=ErrorResponse(
                error=ErrorDetail(
                    message=f"Model '{model}' not found. Available: [{settings.model_name!r}]",
                    type="invalid_request_error",
                    code="model_not_found",
                )
            ).model_dump(),
        )
    return settings.backend_url


def _with_generation_defaults(payload: dict) -> dict:
    extra_body = payload.pop("extra_body", None) or {}
    for key, value in extra_body.items():
        payload.setdefault(key, value)

    payload.setdefault("temperature", settings.default_temperature)
    payload.setdefault("top_p", settings.default_top_p)
    payload.setdefault("presence_penalty", settings.default_presence_penalty)
    payload.setdefault("top_k", settings.default_top_k)
    return payload


async def _proxy_stream(backend_url: str, payload: dict, timeout: float):
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            "POST",
            f"{backend_url}/v1/chat/completions",
            json=payload,
        ) as response:
            if response.status_code != 200:
                body = await response.aread()
                logger.error("Backend error: %s %s", response.status_code, body)
                yield f"data: {json.dumps({'error': 'Backend error'})}\n\n"
                yield "data: [DONE]\n\n"
                return

            async for line in response.aiter_lines():
                if line.strip():
                    yield f"{line}\n\n"
            yield "data: [DONE]\n\n"


@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    backend_url = _get_backend_url(request.model)
    backend_timeout = settings.request_timeout
    payload = _with_generation_defaults(request.model_dump(exclude_none=True))
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    logger.info(
        "Routing %s to %s (stream=%s timeout=%ss)",
        request.model,
        backend_url,
        request.stream,
        backend_timeout,
    )

    if request.stream:
        return StreamingResponse(
            _proxy_stream(backend_url, payload, backend_timeout),
            media_type="text/event-stream",
            headers={"X-Request-Id": request_id},
        )

    async with httpx.AsyncClient(timeout=backend_timeout) as client:
        try:
            response = await client.post(
                f"{backend_url}/v1/chat/completions",
                json=payload,
            )
        except httpx.TimeoutException:
            raise HTTPException(
                status_code=504,
                detail=ErrorResponse(
                    error=ErrorDetail(
                        message=f"Backend request timed out after {backend_timeout:.1f}s",
                        type="server_error",
                        code="timeout",
                    )
                ).model_dump(),
            )
        except httpx.ConnectError:
            raise HTTPException(
                status_code=502,
                detail=ErrorResponse(
                    error=ErrorDetail(
                        message=f"Cannot reach backend for model '{request.model}'",
                        type="server_error",
                        code="backend_unavailable",
                    )
                ).model_dump(),
            )

    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=response.json(),
        )

    result = response.json()
    result["id"] = request_id
    return result
