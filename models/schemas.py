from pydantic import BaseModel, Field
from typing import List


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    subject: str = Field(..., min_length=1)
    grade: int = Field(..., ge=1, le=11)


class SourceChunk(BaseModel):
    page: int
    subject: str
    grade: int
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    confidence: float


class IngestRequest(BaseModel):
    file_path: str = Field(..., min_length=1)
    subject: str = Field(..., min_length=1)
    grade: int = Field(..., ge=1, le=11)


class IngestResponse(BaseModel):
    chunks_indexed: int
    subject: str
    grade: int


class HealthResponse(BaseModel):
    status: str
    qdrant: str
