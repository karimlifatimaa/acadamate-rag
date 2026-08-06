import logging
import time
import requests
from langchain_core.documents import Document
from typing import List, Tuple
from models.schemas import AskResponse, SourceChunk, HistoryTurn
from services.retriever import retrieve
from config import settings

logger = logging.getLogger(__name__)

GENERATE_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama-3.3-70b-versatile"
REQUEST_TIMEOUT = 30


class LLMUnavailableError(Exception):
    """LLM (Groq) cavab vermədikdə — rate limit, şəbəkə və s."""
    pass

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
9. Söhbətin əvvəlki hissəsi verilibsə, sualı onun davamı kimi başa düş (məs. "bəs bu necə olur?" əvvəlki mövzuya aiddir).

Dərslik parçaları:
{context}
{history_block}
Sual: {question}"""


def _build_history_block(history: List[HistoryTurn]) -> str:
    if not history:
        return ""
    lines = [
        f"{'Şagird' if turn.role == 'user' else 'Köməkçi'}: {turn.content}"
        for turn in history[-6:]
    ]
    return "\nSöhbətin əvvəlki hissəsi:\n" + "\n".join(lines) + "\n"


def _build_retrieval_query(question: str, history: List[HistoryTurn]) -> str:
    last_user_turns = [t.content for t in history if t.role == "user"]
    if not last_user_turns:
        return question
    return f"{last_user_turns[-1]}\n{question}"


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
        try:
            response = requests.post(GENERATE_URL, json=payload, headers=headers, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as e:
            logger.warning("Groq şəbəkə xətası", extra={"error": str(e), "attempt": attempt + 1})
            time.sleep(2 ** attempt)
            continue
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

    raise LLMUnavailableError("Groq API cavab vermədi (rate limit və ya əlçatmaz) — bütün cəhdlər uğursuz oldu.")


def ask(question: str, subject: str, grade: int, history: List[HistoryTurn] | None = None) -> AskResponse:
    history = history or []
    logger.info(
        "Ask başladı",
        extra={"subject": subject, "grade": grade, "question_len": len(question), "history_len": len(history)},
    )
    retrieval_query = _build_retrieval_query(question, history)
    results: List[Tuple[Document, float]] = retrieve(retrieval_query, subject, grade)

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
        history_block=_build_history_block(history),
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
