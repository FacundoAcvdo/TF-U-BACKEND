import uuid
from datetime import datetime, timezone
from typing import Any

from chroma_repository import ChromaRepository
from dataset_storage import DatasetStorage
from embeddings import EmbeddingProvider
from models import (
    DatasetDocument,
    DatasetInfo,
    ChromaDocuments,
    Document,
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

        self.supabase = supabase_repository

        self.chroma = chroma_repository

        self.embedding_provider = embedding_provider

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

        dataset_name = (
            self.supabase.get_available_dataset_name(
                user_id=user_id,
                filename=filename,
                user_token=user_token,
            )
        )

        documents = self._parse_documents(
            content=content,
            filename=filename,
            columna_texto=columna_texto,
            columna_clase=columna_clase,
            columna_fecha=columna_fecha,
            formato_original=formato_original,
            formato_detectado=formato_detectado,
        )

        dataset = self._build_dataset_document(
            dataset_id=dataset_id,
            user_id=user_id,
            dataset_name=dataset_name,
            documents=documents,
        )

        chroma_data = self._prepare_chroma_documents(
            dataset_id=dataset_id,
            user_id=user_id,
            dataset_name=dataset_name,
            documents=documents,
        )

        # Cuanto tenga un modelo como la gente genero los vectores y los meto en chroma_data.embeddings

        await self._persist_dataset(
            dataset=dataset,
            chroma_data=chroma_data,
            user_token=user_token,
        )

        return DatasetInfo(
            dataset_id=dataset.dataset_id,
            user_id=dataset.user_id,
            filename=dataset.filename,
            total_documents=len(
                dataset.documents
            ),
            created_at=dataset.created_at,
        )

    def _parse_documents(
        self,
        *,
        content: bytes,
        filename: str,
        columna_texto: str,
        columna_clase: str | None,
        columna_fecha: str | None,
        formato_original: str | None,
        formato_detectado: str | None,
    ) -> list[Document]:

        parser = ParserFactory.get_parser(
            filename
        )

        extra_metadata = {
            "formato_original": formato_original,
            "formato_detectado": formato_detectado,
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

        return documents

    def _build_dataset_document(
        self,
        *,
        dataset_id: str,
        user_id: str,
        dataset_name: str,
        documents: list[Document],
    ) -> DatasetDocument:

        created_at = datetime.now(
            timezone.utc
        ).isoformat()

        return DatasetDocument(
            dataset_id=dataset_id,
            user_id=user_id,
            filename=dataset_name,
            created_at=created_at,
            documents=documents,
        )

    def _prepare_chroma_documents(
        self,
        *,
        dataset_id: str,
        user_id: str,
        dataset_name: str,
        documents: list[Document],
    ) -> ChromaDocuments:

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

        return ChromaDocuments(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
        )

    async def _persist_dataset(
        self,
        *,
        dataset: DatasetDocument,
        chroma_data: ChromaDocuments,
        user_token: str,
    ) -> None:

        try:

            await self.storage.save(
                dataset
            )

            self.chroma.add_documents(
                ids=chroma_data.ids,
                documents=chroma_data.documents,
                metadatas=chroma_data.metadatas,
                embeddings=chroma_data.embeddings,
            )

            await self.supabase.create_dataset(
                dataset_id=dataset.dataset_id,
                user_id=dataset.user_id,
                filename=dataset.filename,
                user_token=user_token,
            )

        except Exception:

            await self._rollback_dataset(
                dataset=dataset,
                chroma_ids=chroma_data.ids,
                user_token=user_token,
            )

            raise

    async def _rollback_dataset(
        self,
        *,
        dataset: DatasetDocument,
        chroma_ids: list[str],
        user_token: str,
    ) -> None:

        try:

            await self.supabase.delete_dataset(
                dataset_id=dataset.dataset_id,
                user_id=dataset.user_id,
                user_token=user_token,
            )

        except Exception:
            pass

        try:

            self.chroma.delete_documents(
                ids=chroma_ids,
            )

        except Exception:
            pass

        try:

            await self.storage.delete(
                user_id=dataset.user_id,
                dataset_id=dataset.dataset_id,
            )

        except Exception:
            pass


    async def get_datasets(
        self,
        *,
        user_id: str,
        user_token: str,
    ) -> list[DatasetInfo]:

        try:
            datasets = await self.supabase.get_datasets_by_user(
                user_id=user_id,
                user_token=user_token,
            )

            return datasets

        except Exception as e:
            print(f"Error retrieving datasets for user {user_id}: {e}")
            raise
