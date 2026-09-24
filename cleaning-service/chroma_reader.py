import asyncio
import os

import chromadb

CHROMA_HOST = os.environ.get("CHROMA_HOST", "chroma-service")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", "8000"))
CHROMA_COLLECTION = os.environ.get("CHROMA_COLLECTION", "documents_e5")
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


def _fetch_sync(where: dict, include: list, page_size: int = PAGE_SIZE) -> dict:
    col = get_collection()
    out: dict = {k: [] for k in include}
    out["ids"] = []
    offset = 0
    while True:
        page = col.get(where=where, include=include, limit=page_size, offset=offset)
        ids = page.get("ids") or []
        if not ids:
            break
        out["ids"].extend(ids)
        for k in include:
            out[k].extend(page.get(k) or [])
        offset += len(ids)
        if len(ids) < page_size:
            break
    return out


async def fetch_paginated(where: dict, include: list, page_size: int = PAGE_SIZE) -> dict:
    return await asyncio.to_thread(_fetch_sync, where, include, page_size)
