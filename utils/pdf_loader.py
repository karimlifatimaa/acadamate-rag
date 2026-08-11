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
# Səhifə başına: bundan az mətn = həmin səhifə şəkil/skandır → OCR lazımdır.
# Qərar hər səhifə üçün AYRI verilir (bütün sənəd üçün tək qərar yox) — çünki
# bəzi kitablarda (məs. qarışıq layout-lu dərsliklər) əksər səhifələr şəkil,
# yalnız bir neçəsi mətn ola bilər; sənəd-səviyyəli cəm bunu gizlədib OCR-i
# səhvən tamam atlaya bilər.
MIN_PAGE_TEXT_THRESHOLD = 40


def _extract_pages(file_path: str) -> List[Tuple[int, str]]:
    """(səhifə_nömrəsi, mətn) siyahısı qaytarır. Mətn qatı olmayan səhifələrə OCR işlədir."""
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    total_pages = len(pages)

    ocr_indices = {i for i, p in enumerate(pages) if len(p.page_content.strip()) < MIN_PAGE_TEXT_THRESHOLD}

    if not ocr_indices:
        logger.info("PDF mətn qatı ilə oxundu", extra={"pages": total_pages})
        return [(p.metadata.get("page", i), p.page_content) for i, p in enumerate(pages)]

    logger.info(
        "OCR lazım olan səhifələr aşkarlandı",
        extra={"pages": total_pages, "ocr_pages": len(ocr_indices)},
    )

    # Səhifə-səhifə emal edirik (hamısını eyni anda yaddaşa yükləmək əvəzinə) —
    # böyük sənədlərdə (100+ səhifə) yaddaş istifadəsini sabit saxlayır, OOM-un
    # qarşısını alır (bütün şəkilləri birdən yükləmək 100+ səhifədə server-in
    # RAM-ını (3-4GB) asanlıqla aşır).
    result: List[Tuple[int, str]] = []
    for i, p in enumerate(pages):
        if i in ocr_indices:
            page_images = convert_from_path(file_path, dpi=OCR_DPI, first_page=i + 1, last_page=i + 1)
            text = pytesseract.image_to_string(page_images[0], lang=OCR_LANG)
            del page_images
        else:
            text = p.page_content
        result.append((p.metadata.get("page", i), text))
        if (i + 1) % 20 == 0 or (i + 1) == total_pages:
            logger.info("Emal gedişatı", extra={"processed": i + 1, "total": total_pages})
    logger.info("PDF emalı tamamlandı", extra={"pages": total_pages, "ocr_pages": len(ocr_indices)})
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
