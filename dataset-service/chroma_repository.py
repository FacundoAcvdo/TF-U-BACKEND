from typing import Any

import chromadb

from config import get_settings


class ChromaRepository:

    def __init__(self) -> None:

        settings = get_settings()

        client_kwargs: dict[str, Any] = {
            "host": settings.chroma_host,
            "port": settings.chroma_port,
        }

        if settings.chroma_token:

            client_kwargs["headers"] = {
                "Authorization": (
                    "Bearer "
                    + settings.chroma_token
                )
            }

        self.client = chromadb.HttpClient(
            **client_kwargs
        )

        self.collection = (
            self.client.get_or_create_collection(
                name=(
                    settings.chroma_collection_name
                )
            )
        )

    def add_documents(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[list[float]] | None = None,
    ) -> None:

        payload: dict[str, Any] = {
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
        }

        if embeddings is not None:
            payload["embeddings"] = embeddings

        self.collection.add(
            **payload
        )

    def delete_documents(
        self,
        ids: list[str],
    ) -> None:

        if not ids:
            return

        self.collection.delete(
            ids=ids,
        )