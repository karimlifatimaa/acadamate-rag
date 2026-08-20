import json
import logging
import re
import requests
from langchain_core.documents import Document
from typing import List, Tuple
from config import settings

logger = logging.getLogger(__name__)

GENERATE_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "openai/gpt-oss-20b"
REQUEST_TIMEOUT = 15
CHUNK_PREVIEW = 500

RERANK_PROMPT = """Sən axtarış nəticələrini qiymətləndirən köməkçisən.

Şagirdin sualı: {question}

Aşağıda nömrələnmiş dərslik parçaları var. Hər parçanın SUALA nə qədər uyğun olduğunu qiymətləndir.

Yalnız sualı HƏQİQƏTƏN cavablandırmağa kömək edən parçaları seç. Mövzu ilə əlaqəsi olmayan parçaları ATMA — onları nəticəyə salma.

Parçalar:
{chunks}

Cavabı YALNIZ JSON massiv kimi qaytar — ən uyğundan ən az uyğuna doğru sıralanmış parça nömrələri.
Heç bir parça uyğun deyilsə, boş massiv qaytar: []
Nümunə cavab: [3, 1, 5]

Cavab:"""


def _parse_indices(text: str, max_index: int) -> List[int]:
    match = re.search(r"\[[\d,\s]*\]", text)
    if not match:
        return []
    try:
        indices = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    # 1-əsaslı nömrələri 0-əsaslı indeksə çevir, sərhədləri yoxla
    result = []
    for i in indices:
        idx = int(i) - 1
        if 0 <= idx < max_index and idx not in result:
            result.append(idx)
    return result


def rerank(question: str, results: List[Tuple[Document, float]], top_k: int) -> List[Tuple[Document, float]]:
    if not results:
        return results

    chunks_text = "\n\n".join(
        f"[{i + 1}] {doc.page_content[:CHUNK_PREVIEW]}"
        for i, (doc, _) in enumerate(results)
    )
    prompt = RERANK_PROMPT.format(question=question, chunks=chunks_text)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.groq_api_key}",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        # 128 reasoning modeli üçün azdır — daxili "düşünmə" tokenləri bunu
        # bitirir, əsl JSON cavaba yer qalmır (content boş qayıdır)
        "max_tokens": 600,
        "temperature": 0.0,
        "reasoning_effort": "low",
    }

    try:
        response = requests.post(GENERATE_URL, json=payload, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        indices = _parse_indices(content, len(results))

        if not indices:
            logger.info("Rerank: uyğun parça tapılmadı", extra={"candidates": len(results)})
            return []

        reranked = [results[i] for i in indices][:top_k]
        logger.info(
            "Rerank tamamlandı",
            extra={"candidates": len(results), "kept": len(reranked)},
        )
        return reranked
    except Exception as e:
        # Rerank uğursuz olsa, orijinal nəticələri qaytar (fallback)
        logger.warning("Rerank uğursuz oldu, orijinal nəticələr qaytarılır", extra={"error": str(e)})
        return results[:top_k]
