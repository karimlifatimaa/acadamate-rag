from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document
from typing import List
from services.embedder import get_embeddings


def load_and_split(file_path: str, subject: str, grade: int, source_key: str = "") -> List[Document]:
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    splitter = SemanticChunker(
        embeddings=get_embeddings(),
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95.0,
    )

    # chunk metadata-da temp path deyil, mənalı açar saxla
    logical_source = source_key or file_path

    chunks: List[Document] = []
    for page in pages:
        page_chunks = splitter.create_documents([page.page_content])
        for chunk in page_chunks:
            if len(chunk.page_content.strip()) < 50:
                continue
            chunk.metadata = {
                "subject": subject,
                "grade": grade,
                "page": page.metadata.get("page", 0),
                "source_file": logical_source,
            }
            chunks.append(chunk)

    return chunks
