import logging
import time
import requests
from langchain_core.documents import Document
from typing import List, Tuple
from models.schemas import AskResponse, SourceChunk
from services.retriever import retrieve
from config import settings

logger = logging.getLogger(__name__)

GENERATE_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama-3.3-70b-versatile"
REQUEST_TIMEOUT = 30

SYSTEM_PROMPT = """Sən Azərbaycan məktəb şagirdlərinə kömək edən müəllim köməkçisisən.
Aşağıdakı dərslik parçalarına əsaslanaraq cavab ver.

Cavab qaydaları:
1. Cavabı {grade}-ci sinif şagirdinə uyğun sadə Azərbaycan dilində izah et.
2. Cavabı DOLĞUN ver — tərif, izah və mümkün olduqda misal göstər.
3. Cavabı strukturlaşdır: əsas tərif → izah → misal(lar). Lazım olarsa siyahı və ya bənd istifadə et.
4. Ən azı 3-5 cümlə yaz. Tək cümləlik cavab vermə.
5. Şagirdin başa düşməsi üçün lazım gələrsə əlaqəli anlayışı da qısaca xatırlat.
6. Şagird müəyyən bir tapşırıq, məşq və ya çalışmanın cavabını soruşursa — tapşırığın mövzusunu dərslik parçalarından istifadə edərək izah et. Tapşırığın özünü tapmağa çalışma, mövzunu aydınlaşdır.
7. Yalnız dərslik parçalarındakı məlumatdan istifadə et, xarici məlumat əlavə etmə.
8. Dərslik parçalarında bu mövzu ilə bağlı heç bir məlumat yoxdursa, "Bu mövzu dərslikdə izah olunmayıb." de.

Dərslik parçaları:
{context}

Sual: {question}"""


def _build_context(docs: List[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page", "?")
        parts.append(f"[Parça {i} — Səhifə {page}]\n{doc.page_content}")
    return "\n\n".join(parts)


def _generate(prompt: str, retries: int = 4) -> str:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.groq_api_key}",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
        "temperature": 0.2,
    }

    for attempt in range(retries):
        response = requests.post(GENERATE_URL, json=payload, headers=headers, timeout=REQUEST_TIMEOUT)
        if response.status_code in (429, 503):
            wait = 2 ** attempt
            logger.warning(
                "Groq retry",
                extra={"status": response.status_code, "wait_s": wait, "attempt": attempt + 1},
            )
            time.sleep(wait)
            continue
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    raise RuntimeError("Groq API yanıt vermədi — bütün cəhdlər uğursuz oldu.")


def ask(question: str, subject: str, grade: int) -> AskResponse:
    logger.info(
        "Ask başladı",
        extra={"subject": subject, "grade": grade, "question_len": len(question)},
    )
    results: List[Tuple[Document, float]] = retrieve(question, subject, grade)

    if not results:
        logger.info("Retrieval boş", extra={"subject": subject, "grade": grade})
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

    logger.info(
        "Ask tamamlandı",
        extra={"chunks": len(sources), "confidence": avg_confidence},
    )
    return AskResponse(
        answer=answer_text,
        sources=sources,
        confidence=avg_confidence,
    )
