import os
import shutil
import tempfile
import logging
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.security.api_key import APIKeyHeader
from models.schemas import IngestResponse
from services.ingestor import ingest_pdf
from config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

ALLOWED_EXTENSIONS = {".pdf"}


def verify_api_key(key: str = Depends(api_key_header)) -> None:
    if key != settings.rag_api_key:
        raise HTTPException(status_code=401, detail="Yanlış API açarı")


@router.post("/ingest", response_model=IngestResponse)
def ingest_endpoint(
    file: UploadFile = File(...),
    subject: str = Form(..., min_length=1),
    grade: int = Form(..., ge=1, le=11),
    _: None = Depends(verify_api_key),
) -> IngestResponse:
    suffix = Path(file.filename or "upload.pdf").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Yalnız PDF faylı qəbul edilir")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        logger.info("PDF yükləndi", extra={"original": file.filename, "tmp": tmp_path})

        count = ingest_pdf(
            file_path=tmp_path,
            subject=subject,
            grade=grade,
            original_name=file.filename or "",
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return IngestResponse(chunks_indexed=count, subject=subject, grade=grade)
