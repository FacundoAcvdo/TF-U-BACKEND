import asyncio
import os

import chromadb
import numpy as np

CHROMA_HOST = os.environ.get("CHROMA_HOST", "chroma-service")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", "8000"))
CHROMA_COLLECTION = os.environ.get("CHROMA_COLLECTION", "documents")
PAGE_SIZE = int(os.environ.get("CHROMA_PAGE_SIZE", "2000"))

_client = None
_collection = None


def get_collection():
    global _client, _collection
    if _collection is not None:
        return _collection
    if _client is None:
        _client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    _collection = _client.get_collection(name=CHROMA_COLLECTION)
    return _collection


def _get_embeddings_sync(dataset_id: str, user_id: str, page_size: int = PAGE_SIZE):
    col = get_collection()
    where = {
        "$and": [
            {"user_id": user_id},
            {"dataset_id": dataset_id},
        ]
    }
    bloques: list = []
    offset = 0
    while True:
        pagina = col.get(where=where, include=["embeddings"], limit=page_size, offset=offset)
        ids = pagina.get("ids") or []
        if not ids:
            break
        emb = pagina.get("embeddings")

        if emb is not None and len(emb) > 0:
            bloques.append(np.asarray(emb, dtype=np.float32))
        offset += len(ids)
        if len(ids) < page_size:
            break
    if not bloques:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(bloques, axis=0)


async def get_dataset_embeddings(dataset_id: str, user_id: str):
    return await asyncio.to_thread(_get_embeddings_sync, dataset_id, user_id)
