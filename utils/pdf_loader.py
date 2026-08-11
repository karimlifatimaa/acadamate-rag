import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document
from typing import List, Tuple
from pdf2image import convert_from_path
import pytesseract
from services.embedder import get_embeddings

logger = logging.getLogger(__name__)

OCR_LANG = "aze"
OCR_DPI = 200
# Bütün sənəd üçün: bundan az mətn = skan-şəkil PDF → OCR lazımdır
MIN_TEXT_THRESHOLD = 100


def _extract_pages(file_path: str) -> List[Tuple[int, str]]:
    """(səhifə_nömrəsi, mətn) siyahısı qaytarır. Mətn qatı yoxdursa OCR işlədir."""
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    total_text = sum(len(p.page_content.strip()) for p in pages)

    # Normal PDF (mətn qatı var) → birbaşa oxu, sürətli
    if total_text >= MIN_TEXT_THRESHOLD:
        logger.info("PDF mətn qatı ilə oxundu", extra={"pages": len(pages), "chars": total_text})
        return [(p.metadata.get("page", i), p.page_content) for i, p in enumerate(pages)]

    # Skan-şəkil PDF (mətn yoxdur) → OCR
    # Səhifə-səhifə emal edirik (hamısını eyni anda yaddaşa yükləmək əvəzinə) —
    # böyük sənədlərdə (100+ səhifə) yaddaş istifadəsini sabit saxlayır, OOM-un
    # qarşısını alır (bütün şəkilləri birdən yükləmək 100+ səhifədə server-in
    # RAM-ını (3-4GB) asanlıqla aşır).
    total_pages = len(pages)
    logger.info("PDF-də mətn qatı yoxdur, OCR başladı", extra={"pages": total_pages})
    result: List[Tuple[int, str]] = []
    for page_num in range(1, total_pages + 1):
        page_images = convert_from_path(file_path, dpi=OCR_DPI, first_page=page_num, last_page=page_num)
        text = pytesseract.image_to_string(page_images[0], lang=OCR_LANG)
        result.append((page_num - 1, text))
        del page_images
        if page_num % 20 == 0 or page_num == total_pages:
            logger.info("OCR gedişat", extra={"processed": page_num, "total": total_pages})
    logger.info("OCR tamamlandı", extra={"pages": total_pages})
    return result


def load_and_split(file_path: str, subject: str, grade: int, source_key: str = "") -> List[Document]:
    pages = _extract_pages(file_path)

    splitter = SemanticChunker(
        embeddings=get_embeddings(),
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95.0,
    )

    # chunk metadata-da temp path deyil, mənalı açar saxla
    logical_source = source_key or file_path

    chunks: List[Document] = []
    for page_num, page_text in pages:
        if not page_text.strip():
            continue
        page_chunks = splitter.create_documents([page_text])
        for chunk in page_chunks:
            if len(chunk.page_content.strip()) < 50:
                continue
            chunk.metadata = {
                "subject": subject,
                "grade": grade,
                "page": page_num,
                "source_file": logical_source,
            }
            chunks.append(chunk)

    return chunks
