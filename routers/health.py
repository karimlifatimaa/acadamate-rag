from fastapi import APIRouter
from models.schemas import HealthResponse
from services.vector_store import get_qdrant_client

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        client = get_qdrant_client()
        client.get_collections()
        qdrant_status = "up"
    except Exception:
        qdrant_status = "down"

    return HealthResponse(status="ok", qdrant=qdrant_status)
