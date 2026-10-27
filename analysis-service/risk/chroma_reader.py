"""Acceso a Chroma con cliente reutilizado + paginado + to_thread.

Copia del patrón de emotions-service: pide documents+metadatas+embeddings.
"""
import asyncio
import os

import chromadb

CHROMA_HOST = os.environ.get("CHROMA_HOST", "chroma-service")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", "8000"))
CHROMA_COLLECTION = os.environ.get("CHROMA_COLLECTION", "documents")
CHROMA_CHUNKED_COLLECTION = os.environ.get("CHROMA_CHUNKED_COLLECTION", "chunked_chats")
PAGE_SIZE = int(os.environ.get("CHROMA_PAGE_SIZE", "2000"))

_client = None
_collections: dict = {}


def get_collection(name: str | None = None):
    global _client
    name = name or CHROMA_COLLECTION
    if name in _collections:
        return _collections[name]
    if _client is None:
        _client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    _collections[name] = _client.get_collection(name=name)
    return _collections[name]


def _fetch_sync(where: dict, include: list, collection: str | None = None,
                page_size: int = PAGE_SIZE) -> dict:
    col = get_collection(collection)
    out: dict = {k: [] for k in include}
    out["ids"] = []
    offset = 0
    while True:
        page = col.get(where=where, include=include, limit=page_size, offset=offset)
        raw_ids = page.get("ids")
        ids = list(raw_ids) if raw_ids is not None else []
        if not ids:
            break
        out["ids"].extend(ids)
        for k in include:
            vals = page.get(k)
            if vals is not None:
                out[k].extend(list(vals))
        offset += len(ids)
        if len(ids) < page_size:
            break
    return out


async def fetch_paginated(where: dict, include: list, collection: str | None = None,
                        page_size: int = PAGE_SIZE) -> dict:
    return await asyncio.to_thread(_fetch_sync, where, include, collection, page_size)
