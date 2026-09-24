import asyncio
import os

import chromadb

from config import get_settings


class ChromaService:

    def __init__(self) -> None:

        settings = get_settings()

        self.client = chromadb.HttpClient(
            host=settings.chroma_host or os.environ.get("CHROMA_HOST", "chroma-service"),
            port=settings.chroma_port or int(os.environ.get("CHROMA_PORT", "8000")),
        )
        self._collection = None
        self._page_size = int(os.environ.get("CHROMA_PAGE_SIZE", "2000"))

    def _get_collection(self):
        # Reutiliza collection (no pedirla por request).
        if self._collection is None:
            self._collection = self.client.get_collection(
                name=get_settings().chroma_collection_name
            )
        return self._collection

    def _fetch_sync(self, *, dataset_id: str, user_id: str) -> dict:
        collection = self._get_collection()
        where = {
            "$and": [
                {"dataset_id": dataset_id},
                {"user_id": user_id},
            ]
        }
        include = ["documents", "embeddings", "metadatas"]
        out: dict = {"ids": [], "documents": [], "embeddings": [], "metadatas": []}
        offset = 0
        while True:
            page = collection.get(
                where=where, include=include,
                limit=self._page_size, offset=offset,
            )
            ids = page.get("ids") or []
            if not ids:
                break
            out["ids"].extend(ids)
            out["documents"].extend(page.get("documents") or [])
            out["embeddings"].extend(page.get("embeddings") or [])
            out["metadatas"].extend(page.get("metadatas") or [])
            offset += len(ids)
            if len(ids) < self._page_size:
                break
        return out

    async def get_dataset_documents(
        self,
        *,
        dataset_id: str,
        user_id: str,
    ):
        # HttpClient es bloqueante → to_thread.
        return await asyncio.to_thread(
            self._fetch_sync, dataset_id=dataset_id, user_id=user_id
        )

