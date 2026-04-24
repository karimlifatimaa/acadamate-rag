import os
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security.api_key import APIKeyHeader
from models.schemas import IngestRequest, IngestResponse
from services.ingestor import ingest_pdf
from config import settings

router = APIRouter()

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(key: str = Depends(api_key_header)) -> None:
    if key != settings.rag_api_key:
        raise HTTPException(status_code=401, detail="Yanlış API açarı")


@router.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(
    request: IngestRequest,
    _: None = Depends(verify_api_key),
) -> IngestResponse:
    if not os.path.isfile(request.file_path):
        raise HTTPException(status_code=404, detail=f"Fayl tapılmadı: {request.file_path}")

    count = ingest_pdf(
        file_path=request.file_path,
        subject=request.subject,
        grade=request.grade,
    )
    return IngestResponse(chunks_indexed=count, subject=request.subject, grade=request.grade)
