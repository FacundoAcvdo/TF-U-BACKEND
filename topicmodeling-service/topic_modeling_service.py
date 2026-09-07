from typing import Any
from chroma_service import ChromaService
from storage_service import StorageService
from supabase_service import SupabaseService
from topic_modeling_provider import TopicModelingProvider


class TopicModelingService:

    def __init__(
        self,
        supabase_repository: SupabaseService,
        chroma_repository: ChromaService,
        topic_provider: TopicModelingProvider,
        storage_service: StorageService,
    ) -> None:
        self.supabase = supabase_repository
        self.chroma = chroma_repository
        self.topic_provider = topic_provider
        self.storage = storage_service

    async def get_topics(
        self,
        *,
        dataset_name: str,
        user_id: str,
        user_token: str,
    ) -> dict[str, Any]:

        dataset = await self.supabase.get_dataset_by_name(
            dataset_name=dataset_name,
            user_id=user_id,
            user_token=user_token,
        )

        if not dataset:
            raise LookupError(f"No se encontró el dataset '{dataset_name}'.")

        dataset_id = dataset["dataset_id"]

        cached_result = await self.storage.get_topics(
            user_id=user_id,
            dataset_id=dataset_id,
        )

        if cached_result is not None:
            return cached_result

        chroma_data = await self.chroma.get_dataset_documents(
            dataset_id=dataset_id,
            user_id=user_id,
        )

        documents = chroma_data["documents"]
        embeddings = chroma_data["embeddings"]
        document_ids = chroma_data["ids"]

        if documents is None or len(documents) == 0:
            raise ValueError(
                "El dataset no contiene documentos para realizar topic modeling."
            )

        if embeddings is None or len(embeddings) == 0:
            raise ValueError("El dataset no contiene embeddings.")

        if len(documents) != len(embeddings):
            raise ValueError(
                "La cantidad de documentos no coincide con la cantidad de embeddings."
            )

        resultado = await self.topic_provider.extract_topics(
            documents=documents,
            embeddings=embeddings,
            document_ids=document_ids,
        )

        await self.storage.save_topics(
            user_id=user_id,
            dataset_id=dataset_id,
            data=resultado,
        )

        return resultado