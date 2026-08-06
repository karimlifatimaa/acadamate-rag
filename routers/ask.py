from fastapi import APIRouter, Depends, HTTPException
from fastapi.security.api_key import APIKeyHeader
from models.schemas import AskRequest, AskResponse
from services.generator import ask, LLMUnavailableError
from config import settings

router = APIRouter()

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(key: str = Depends(api_key_header)) -> None:
    if key != settings.rag_api_key:
        raise HTTPException(status_code=401, detail="Yanlış API açarı")


@router.post("/ask", response_model=AskResponse)
def ask_endpoint(
    request: AskRequest,
    _: None = Depends(verify_api_key),
) -> AskResponse:
    try:
        return ask(
            question=request.question,
            subject=request.subject,
            grade=request.grade,
            history=request.history,
        )
    except LLMUnavailableError:
        raise HTTPException(
            status_code=503,
            detail="AI xidməti hazırda məşğuldur (sorğu limiti). Bir neçə saniyədən sonra yenidən cəhd edin.",
        )
