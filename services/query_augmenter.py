import logging
import requests
from config import settings

logger = logging.getLogger(__name__)

GENERATE_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "openai/gpt-oss-20b"
REQUEST_TIMEOUT = 10

HYDE_PROMPT = """Sən Azərbaycan məktəb dərsliklərindən axtarış üçün sual hazırlayan köməkçisən.

Addımlar:
1. Şagirdin sualını diqqətlə oxu.
2. Sual bir tapşırıq, məşq, çalışma və ya səhifəyə istinad edirsə — o tapşırığın NƏ MÖVZUSU OLDUĞUNU müəyyən et. Tapşırığın cavabını deyil, mövzusunu izah et.
3. Mövzu haqqında dərslik üslubunda 2-3 cümləlik fərzi mətn yaz.

Nümunələr:
Sual: "səhifə 36 tapşırıq d: -şünas şəkilçisinin funksiyasını izah et"
Fərzi cavab: "-şünas şəkilçisi sözlərə qoşularaq həmin sahənin bilicisi, mütəxəssisi mənasını bildirir. Şərqşünas, misirşünas sözlərindəki kimi bu şəkilçi elmi ixtisas bildirir. -çı4 şəkilçisindən fərqi odur ki, -şünas daha dar, elmi mənalı peşə ifadə edir."

Sual: "pifaqor teoremi nədir səh 45"
Fərzi cavab: "Pifaqor teoremi düzbucaqlı üçbucaqda hipotenuzun kvadratı iki katetin kvadratları cəminə bərabərdir."

Yalnız fərzi mətni qaytar — izahatsız, başlıqsız.

Şagird sualı: {question}

Fərzi dərslik mətni:"""


def augment_query(question: str) -> str:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.groq_api_key}",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": HYDE_PROMPT.format(question=question)}],
        "max_tokens": 500,
        "temperature": 0.3,
        "reasoning_effort": "low",
    }

    try:
        response = requests.post(GENERATE_URL, json=payload, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        hyde_answer = response.json()["choices"][0]["message"]["content"].strip()
        logger.info("HyDE augmented", extra={"original_len": len(question), "hyde_len": len(hyde_answer)})
        return f"{question}\n{hyde_answer}"
    except Exception as e:
        logger.warning("Query augmentation uğursuz oldu", extra={"error": str(e)})
        return question
