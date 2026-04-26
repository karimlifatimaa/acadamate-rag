import requests
from config import settings

GENERATE_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama-3.3-70b-versatile"

HYDE_PROMPT = """Sən şagird sualını Azərbaycan dərsliklərində axtarış üçün hazırlayan köməkçisən.

Tapşırıq:
1. Şagirdin sualını ədəbi Azərbaycan dilinə çevir (ləhcə, qısaltma, yazı səhvlərini düzəlt).
2. Sonra dərslik üslubunda 2-3 cümləlik FƏRZİ cavab yaz.

Yalnız fərzi cavabı qaytar — izahatsız, başlıqsız.

Şagird sualı: {question}

Fərzi dərslik cavabı:"""


def augment_query(question: str) -> str:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.groq_api_key}",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": HYDE_PROMPT.format(question=question)}],
        "max_tokens": 256,
        "temperature": 0.3,
    }

    try:
        response = requests.post(GENERATE_URL, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        hyde_answer = response.json()["choices"][0]["message"]["content"].strip()
        return f"{question}\n{hyde_answer}"
    except Exception as e:
        print(f"Query augmentation uğursuz oldu: {e}")
        return question
