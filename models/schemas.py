from pydantic import BaseModel, Field
from typing import List, Optional


class HistoryTurn(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1, max_length=2000)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    subject: str = Field(..., min_length=1)
    grade: int = Field(..., ge=1, le=11)
    history: List[HistoryTurn] = Field(default_factory=list, max_length=6)


class SourceChunk(BaseModel):
    page: int
    subject: str
    grade: int
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    confidence: float


class IngestResponse(BaseModel):
    chunks_indexed: int
    subject: str
    grade: int


class HealthResponse(BaseModel):
    status: str
    qdrant: str


class BookInfo(BaseModel):
    id: str
    subject: str
    grade: int
    source_file: str
    original_name: str
    uploaded_at: str
    status: str
    total_chunks: int


class BooksResponse(BaseModel):
    books: List[BookInfo]
    total: int


class DeleteBooksResponse(BaseModel):
    deleted_chunks: int
