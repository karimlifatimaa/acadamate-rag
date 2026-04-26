from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.security.api_key import APIKeyHeader
from typing import Optional
from models.schemas import BooksResponse, DeleteBooksResponse
from services.books import list_books, delete_book_by_id, delete_books
from config import settings

router = APIRouter()

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(key: str = Depends(api_key_header)) -> None:
    if key != settings.rag_api_key:
        raise HTTPException(status_code=401, detail="Yanlış API açarı")


@router.get("/books", response_model=BooksResponse)
def list_books_endpoint(_: None = Depends(verify_api_key)) -> BooksResponse:
    books = list_books()
    return BooksResponse(books=books, total=len(books))


@router.delete("/books/{book_id}", response_model=DeleteBooksResponse)
def delete_book_endpoint(
    book_id: str,
    _: None = Depends(verify_api_key),
) -> DeleteBooksResponse:
    deleted = delete_book_by_id(book_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Kitab tapılmadı")
    return DeleteBooksResponse(deleted_chunks=deleted)


@router.delete("/books", response_model=DeleteBooksResponse)
def delete_books_by_filter_endpoint(
    subject: Optional[str] = Query(None, min_length=1),
    grade: Optional[int] = Query(None, ge=1, le=11),
    source_file: Optional[str] = Query(None, min_length=1),
    _: None = Depends(verify_api_key),
) -> DeleteBooksResponse:
    if subject is None and grade is None and source_file is None:
        raise HTTPException(
            status_code=400,
            detail="Ən azı bir filter təyin et: subject, grade və ya source_file",
        )
    deleted = delete_books(subject=subject, grade=grade, source_file=source_file)
    return DeleteBooksResponse(deleted_chunks=deleted)
