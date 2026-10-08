from typing import Any, Optional
from chroma_service import ChromaService
from supabase_service import SupabaseService
from title_provider import TitleProvider
from topic_modeling_provider import TopicModelingProvider


class TopicModelingService:

    def __init__(
        self,
        supabase_repository: SupabaseService,
        chroma_repository: ChromaService,
        topic_provider: TopicModelingProvider,
        title_provider: Optional[TitleProvider] = None,
    ) -> None:
        self.supabase = supabase_repository
        self.chroma = chroma_repository
        self.topic_provider = topic_provider
        self.title_provider = title_provider

    async def get_topics(
        self,
        *,
        dataset_name: str,
        user_id: str,
        user_token: str,
        max_topics: int=30,
    ) -> dict[str, Any]:

        dataset = await self.supabase.get_dataset_by_name(
            dataset_name=dataset_name,
            user_id=user_id,
            user_token=user_token,
        )

        if not dataset:
            raise LookupError(f"No se encontró el dataset '{dataset_name}'.")

        dataset_id = dataset["dataset_id"]

        cached_result = await self.supabase.get_topic_results(
            user_id=user_id,
            dataset_id=dataset_id,
            max_topics=max_topics,
            user_token=user_token,
        )

        if cached_result is not None:
            enriched = await self._ensure_titles(
                resultado=cached_result,
                user_id=user_id,
                dataset_id=dataset_id,
                max_topics=max_topics,
                user_token=user_token,
            )
            return enriched

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
            max_topics=max_topics,
        )

        resultado = await self._apply_titles(resultado)

        try:
            await self.supabase.save_topic_results(
                user_id=user_id,
                dataset_id=dataset_id,
                max_topics=max_topics,
                resultado=resultado,
                user_token=user_token,
            )
        except Exception as exc:
            print(
                "[TopicModelingService] No se pudo guardar en "
                f"Supabase: {exc}"
            )

        return resultado

    async def get_cached_topics(
        self,
        *,
        dataset_name: str,
        user_id: str,
        user_token: str,
        max_topics: int = 30,
    ) -> dict[str, Any] | None:
        """Solo lee Supabase. No calcula ni llama a Gemini."""
        dataset = await self.supabase.get_dataset_by_name(
            dataset_name=dataset_name,
            user_id=user_id,
            user_token=user_token,
        )

        if not dataset:
            raise LookupError(f"No se encontró el dataset '{dataset_name}'.")

        return await self.supabase.get_topic_results(
            user_id=user_id,
            dataset_id=dataset["dataset_id"],
            max_topics=max_topics,
            user_token=user_token,
        )

    async def _apply_titles(
        self,
        resultado: dict[str, Any],
    ) -> dict[str, Any]:
        topics = resultado.get("topics", [])

        if self.title_provider is None:
            for topic in topics:
                topic.setdefault(
                    "title",
                    TitleProvider.fallback_title(topic.get("keywords", [])),
                )
                topic.setdefault("title_source", "keywords")
            return resultado

        titles = await self.title_provider.generate_titles(topics)

        for topic in topics:
            topic_id = int(topic["topic_id"])
            if topic_id in titles:
                topic["title"] = titles[topic_id]
                topic["title_source"] = "llm"
            else:
                topic["title"] = TitleProvider.fallback_title(
                    topic.get("keywords", [])
                )
                topic["title_source"] = (
                    "keywords"
                    if self.title_provider.is_available
                    else "disabled"
                )

        return resultado

    async def _ensure_titles(
        self,
        *,
        resultado: dict[str, Any],
        user_id: str,
        dataset_id: str,
        max_topics: int,
        user_token: str,
    ) -> dict[str, Any]:
        """Etiqueta cachés viejos que no tienen `title` y los re-graba."""
        topics = resultado.get("topics", [])

        if topics and all("title" in topic for topic in topics):
            return resultado

        updated = await self._apply_titles(resultado)

        try:
            await self.supabase.save_topic_results(
                user_id=user_id,
                dataset_id=dataset_id,
                max_topics=max_topics,
                resultado=updated,
                user_token=user_token,
            )
        except Exception as exc:
            print(f"[TopicModelingService] No se pudo re-grabar caché con títulos: {exc}")

        return updated