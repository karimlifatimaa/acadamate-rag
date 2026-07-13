import logging
from langchain_openai import AzureOpenAIEmbeddings
from config import settings

logger = logging.getLogger(__name__)


def get_embeddings() -> AzureOpenAIEmbeddings:
    logger.info(
        "Azure OpenAI embedding istifadə olunur",
        extra={"deployment": settings.azure_openai_deployment},
    )
    return AzureOpenAIEmbeddings(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        azure_deployment=settings.azure_openai_deployment,
        api_version=settings.azure_openai_api_version,
    )
