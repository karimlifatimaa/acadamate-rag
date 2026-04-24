from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List


def load_and_split(file_path: str, subject: str, grade: int) -> List[Document]:
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "!", "?", " "],
    )
    chunks = splitter.split_documents(pages)

    for chunk in chunks:
        chunk.metadata["subject"] = subject
        chunk.metadata["grade"] = grade
        chunk.metadata["source_file"] = file_path
        # PyPDFLoader artıq page metadata-sını əlavə edir (page key)

    return chunks
