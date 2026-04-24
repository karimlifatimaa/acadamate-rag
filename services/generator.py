import requests
from langchain_core.documents import Document
from typing import List, Tuple
from models.schemas import AskResponse, SourceChunk
from services.retriever import retrieve
from config import settings

GENERATE_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-flash-latest:generateContent"
)

SYSTEM_PROMPT = """Sən Azərbaycan məktəb şagirdlərinə kömək edən müəllim köməkçisisən.
Aşağıdakı dərslik parçalarına YALNIZ əsaslanaraq cavab ver.
Cavabı {grade}-ci sinif şagirdinə uyğun sadə Azərbaycan dilində izah et.
Əgər cavab dərslik parçalarında yoxdursa, "Bu mövzu dərslikdə izah olunmayıb." de.
Xarici məlumat əlavə etmə. Yalnız dərslik mövzusunda cavab ver.

Dərslik parçaları:
{context}

Sual: {question}"""


def _build_context(docs: List[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page", "?")
        parts.append(f"[Parça {i} — Səhifə {page}]\n{doc.page_content}")
    return "\n\n".join(parts)


def _generate(prompt: str) -> str:
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": settings.google_api_key,
    }
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 1024, "temperature": 0.2},
    }
    response = requests.post(GENERATE_URL, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()["candidates"][0]["content"]["parts"][0]["text"]


def ask(question: str, subject: str, grade: int) -> AskResponse:
    results: List[Tuple[Document, float]] = retrieve(question, subject, grade)

    if not results:
        return AskResponse(
            answer="Bu mövzu dərslikdə izah olunmayıb.",
            sources=[],
            confidence=0.0,
        )

    docs = [doc for doc, _ in results]
    scores = [score for _, score in results]
    context = _build_context(docs)
    avg_confidence = round(sum(scores) / len(scores), 2)

    prompt = SYSTEM_PROMPT.format(
        grade=grade,
        context=context,
        question=question,
    )

    answer_text = _generate(prompt)

    sources = [
        SourceChunk(
            page=doc.metadata.get("page", 0),
            subject=doc.metadata.get("subject", subject),
            grade=doc.metadata.get("grade", grade),
            excerpt=doc.page_content[:200],
        )
        for doc in docs
    ]

    return AskResponse(
        answer=answer_text,
        sources=sources,
        confidence=avg_confidence,
    )
