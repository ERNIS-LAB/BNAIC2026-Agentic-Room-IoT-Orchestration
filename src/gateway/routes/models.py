from fastapi import APIRouter

from gateway.config import settings
from gateway.schemas import ModelListResponse, ModelObject

router = APIRouter()


@router.get("/v1/models", response_model=ModelListResponse)
async def list_models():
    return ModelListResponse(
        data=[
            ModelObject(id=settings.model_name),
        ]
    )
