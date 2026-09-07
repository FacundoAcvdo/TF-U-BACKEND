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
        batch_size: int = 2000,
    ) -> None:
        total = len(ids)

        if not (len(documents) == total == len(metadatas)):
            raise ValueError(
                f"Error de dimensiones: ids={total}, documents={len(documents)}, metadatas={len(metadatas)}"
            )

        if embeddings is not None and len(embeddings) != total:
            raise ValueError(
                f"Error de dimensiones en embeddings: esperados {total}, recibidos {len(embeddings)}"
            )

        for i in range(0, total, batch_size):
            fin = min(i + batch_size, total)

            batch_ids = ids[i:fin]
            batch_docs = documents[i:fin]
            batch_metas = metadatas[i:fin]

            sanitized_metas = []
            for m in batch_metas:
                meta_valida = {k: v for k, v in m.items() if v is not None} if isinstance(m, dict) else {}
                if not meta_valida:
                    meta_valida = {"_v": 1}
                sanitized_metas.append(meta_valida)

            payload: dict[str, Any] = {
                "ids": batch_ids,
                "documents": batch_docs,
                "metadatas": sanitized_metas,
            }

            if embeddings is not None:
                payload["embeddings"] = embeddings[i:fin]

            # Inserción en Chroma
            self.collection.add(**payload)

    def delete_documents(
        self,
        ids: list[str],
    ) -> None:

        if not ids:
            return

        self.collection.delete(
            ids=ids,
        )