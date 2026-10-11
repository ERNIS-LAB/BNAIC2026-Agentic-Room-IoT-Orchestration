import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from gateway.config import settings
from gateway.routes import agent, chat, models
from gateway.services.iot_runtime import close_iot_runtime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def _check_backend(name: str, url: str) -> dict:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{url}/health")
            healthy = response.status_code == 200
    except Exception:
        healthy = False
    return {"name": name, "url": url, "healthy": healthy}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Gateway starting on port %d", settings.gateway_port)
    logger.info("Multimodal backend: %s", settings.backend_url)

    status = await _check_backend(settings.model_name, settings.backend_url)
    if status["healthy"]:
        logger.info("Backend '%s' is healthy", settings.model_name)
        # Prime the LLM prompt cache with a minimal completion so the first
        # real agent query doesn't pay the cold-start tax.
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                warmup_resp = await client.post(
                    f"{settings.backend_url.rstrip('/')}/v1/chat/completions",
                    json={
                        "model": settings.model_name,
                        "messages": [{"role": "user", "content": "OK"}],
                        "max_tokens": 1,
                        "temperature": 0.0,
                    },
                )
                if warmup_resp.status_code == 200:
                    logger.info(
                        "LLM prompt cache warmed up (status %s)",
                        warmup_resp.status_code,
                    )
                else:
                    logger.warning(
                        "LLM warmup returned status %s", warmup_resp.status_code
                    )
        except Exception:
            logger.warning(
                "LLM warmup request failed; first query may be slower", exc_info=True
            )
    else:
        logger.warning(
            "Backend '%s' is NOT reachable at %s",
            settings.model_name,
            settings.backend_url,
        )

    try:
        yield
    finally:
        close_iot_runtime()


app = FastAPI(
    title="LLM Gateway",
    description="OpenAI-compatible API gateway for local multimodal LLM inference",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(models.router)
app.include_router(agent.router)


@app.get("/health")
async def health():
    backends = [
        await _check_backend(settings.model_name, settings.backend_url),
    ]
    all_healthy = all(b["healthy"] for b in backends)
    return {
        "status": "healthy" if all_healthy else "degraded",
        "backends": backends,
    }
