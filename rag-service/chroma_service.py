import os
import chromadb
from chromadb.config import Settings
import logging
from logging_utils import setup_service_logger
logger = setup_service_logger("rag-service")


CHROMA_HOST = os.environ.get("CHROMA_HOST", "chroma-service")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", "8000"))
COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION", "rag_docs")


CHROMA_FALLBACK_LOCAL = os.environ.get("CHROMA_FALLBACK_LOCAL", "false").lower() in ("true", "1", "yes")

def get_chroma_client():
    try:
        client = chromadb.HttpClient(
            host=CHROMA_HOST,
            port=CHROMA_PORT,
            settings=Settings(anonymized_telemetry=False)
        )
        client.heartbeat()
        return client
    except Exception as e:
        if CHROMA_FALLBACK_LOCAL:
            logger.exception(f"[chroma_service] No se pudo conectar a HttpClient {CHROMA_HOST}:{CHROMA_PORT} -> {e}")
            logger.warning("[chroma_service] Fallback a PersistentClient local ./chroma_data")
            return chromadb.PersistentClient(path="./chroma_data", settings=Settings(anonymized_telemetry=False))
        logger.exception(f"[chroma_service] ERROR conectando a Chroma {CHROMA_HOST}:{CHROMA_PORT} -> {e}")
        raise


def get_or_create_collection(client=None, name: str | None = None):
    if client is None:
        client = get_chroma_client()

    collection_name = name or COLLECTION_NAME
    try:
        collection = client.get_collection(name=collection_name)
        pass
        return collection
    except Exception:
        pass
        try:
            collection = client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            logger.exception(f"[chroma_service] Error creando colección: {e}")
            collection = client.get_collection(name=collection_name)
        return collection
