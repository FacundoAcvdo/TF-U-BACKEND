import uuid
from datetime import datetime, timezone
from typing import Any

from chroma_repository import ChromaRepository
from dataset_storage import DatasetStorage
from embeddings import EmbeddingProvider
from models import (
    DatasetDocument,
    DatasetInfo,
)
from parsers import ParserFactory
from supabase_repository import (
    SupabaseRepository,
)


class DatasetService:

    def __init__(
        self,
        dataset_storage: DatasetStorage,
        supabase_repository: SupabaseRepository,
        chroma_repository: ChromaRepository,
        embedding_provider: EmbeddingProvider,
    ) -> None:

        self.storage = dataset_storage

        self.supabase = (supabase_repository)

        self.chroma = chroma_repository

        self.embedding_provider = (embedding_provider)

    async def create_dataset(
        self,
        *,
        content: bytes,
        filename: str,
        columna_texto: str,
        columna_clase: str | None,
        columna_fecha: str | None,
        formato_original: str | None,
        formato_detectado: str | None,
        user_id: str,
        user_token: str,
    ) -> DatasetInfo:

        dataset_id = str(
            uuid.uuid4()
        )

        created_at = (
            datetime.now(
                timezone.utc
            ).isoformat()
        )

        dataset_name = self.supabase.get_available_dataset_name(
            user_id=user_id,
            filename=filename,
            user_token=user_token
        )

        parser = ParserFactory.get_parser(
            filename
        )   

        extra_metadata = {
            "formato_original": (
                formato_original
            ),
            "formato_detectado": (
                formato_detectado
            ),
        }

        extra_metadata = {
            key: value
            for key, value
            in extra_metadata.items()
            if value is not None
        }

        documents = parser.parse(
            content=content,
            columna_texto=columna_texto,
            columna_clase=columna_clase,
            columna_fecha=columna_fecha,
            extra_metadata=extra_metadata,
        )

        if not documents:

            raise ValueError(
                "El archivo no contiene "
                "texto procesable."
            )

        dataset_document = DatasetDocument(
            dataset_id=dataset_id,
            user_id=user_id,
            filename=dataset_name,
            created_at=created_at,
            documents=documents,
        )

        await self.storage.save(
            dataset_document
        )

        ids: list[str] = []

        texts: list[str] = []

        metadatas: list[
            dict[str, Any]
        ] = []

        for document in documents:

            document_id = str(
                uuid.uuid4()
            )

            metadata = (
                document.to_chroma_metadata()
            )

            metadata.update(
                {
                    "dataset_id": dataset_id,
                    "user_id": user_id,
                    "source_filename": dataset_name,
                }
            )

            ids.append(
                document_id
            )

            texts.append(
                document.texto
            )

            metadatas.append(
                metadata
            )

        try:
            print("generando embeddings")
            self.chroma.add_documents(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
            )

        except Exception:

            raise

        print("guardado en Supabase")

        try:

            await self.supabase.create_dataset(
                dataset_id=dataset_id,
                user_id=user_id,
                filename=dataset_name,
                user_token=user_token
            )

        except Exception:

            raise

        return DatasetInfo(
            dataset_id=dataset_id,
            user_id=user_id,
            filename=dataset_name,
            total_documents=len(
                documents
            ),
            created_at=created_at,
        )

    async def get_datasets(
        self,
        *,
        user_id: str,
        user_token: str,
    ) -> list[DatasetInfo]:

        supabase = self.supabase.get_supabase_client(user_jwt=user_token)

        response = (
            supabase.table("datasets")
            .select("*")
            .eq("user_id", user_id)
            .execute()
        )

        datasets = response.data

        return datasets
