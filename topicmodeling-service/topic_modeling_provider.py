from typing import Any
import asyncio
import numpy as np
from bertopic import BERTopic


class TopicModelingProvider:

    def __init__(self) -> None:
        self.model_config = {
            "verbose": False,
        }

    async def extract_topics(
        self,
        *,
        documents: list[str],
        embeddings: list[list[float]],
        document_ids: list[str],
    ) -> dict[str, Any]:
        """
        Ejecuta un análisis independiente de BERTopic
        sobre un único dataset.

        Cada llamada crea su propia instancia de BERTopic.
        """
        return await asyncio.to_thread(
            self._extract_topics_sync,
            documents,
            embeddings,
            document_ids,
        )

    def _extract_topics_sync(
        self,
        documents: list[str],
        embeddings: list[list[float]],
        document_ids: list[str],
    ) -> dict[str, Any]:

        print(f"Ejecutando topic modeling con {len(documents)} documentos.")

        if documents is None or len(documents) == 0:
            print("[ERROR] No hay documentos para realizar topic modeling.")
            raise ValueError(
                "No hay documentos para realizar topic modeling."
            )

        if embeddings is None or len(embeddings) == 0:
            print("[ERROR] No hay embeddings para realizar topic modeling.")
            raise ValueError(
                "No hay embeddings para realizar topic modeling."
            )

        if len(documents) != len(embeddings):
            print(
                f"[ERROR] La cantidad de documentos ({len(documents)}) "
                f"y embeddings ({len(embeddings)}) no coincide."
            )
            raise ValueError(
                "La cantidad de documentos y embeddings no coincide."
            )

        if len(documents) != len(document_ids):
            print(
                f"[ERROR] La cantidad de documentos ({len(documents)}) "
                f"e IDs ({len(document_ids)}) no coincide."
            )
            raise ValueError(
                "La cantidad de documentos e IDs no coincide."
            )

        embeddings_np = np.asarray(embeddings, dtype=np.float32)

        model = BERTopic(
            **self.model_config,
        )

        topics_raw, probabilities = model.fit_transform(
            documents,
            embeddings=embeddings_np,
        )

        topics = [int(t) for t in topics_raw]
        document_count = len(documents)

        outlier_count = topics.count(-1)

        outlier_percentage = (
            (outlier_count / document_count) * 100
            if document_count > 0
            else 0
        )

        unique_topics = sorted(set(topics))

        valid_topics = [
            topic_id
            for topic_id in unique_topics
            if topic_id != -1
        ]

        topic_results = []

        for topic_id in valid_topics:

            topic_indexes = [
                index
                for index, assigned_topic in enumerate(topics)
                if assigned_topic == topic_id
            ]

            topic_document_count = len(topic_indexes)

            percentage = (
                topic_document_count
                / document_count
                * 100
            )

            topic_words = model.get_topic(topic_id)

            if topic_words is None:
                keywords = []
            else:
                keywords = [
                    word
                    for word, score in topic_words[:10]
                ]

            representative_documents = (
                self._get_representative_documents(
                    model=model,
                    topic_id=topic_id,
                    documents=documents,
                    document_ids=document_ids,
                    topic_indexes=topic_indexes,
                )
            )

            topic_results.append(
                {
                    "topic_id": topic_id,
                    "document_count": topic_document_count,
                    "percentage": round(percentage, 2),
                    "keywords": keywords,
                    "representative_documents": representative_documents,
                }
            )

        topic_results.sort(
            key=lambda topic: topic["document_count"],
            reverse=True,
        )

        return {
            "document_count": document_count,
            "topic_count": len(topic_results),
            "outlier_percentage": round(outlier_percentage, 2),
            "topics": topic_results,
        }

    def _get_representative_documents(
        self,
        *,
        model: BERTopic,
        topic_id: int,
        documents: list[str],
        document_ids: list[str],
        topic_indexes: list[int],
        limit: int = 3,
    ) -> list[dict[str, str]]:
        """
        Obtiene documentos representativos del topic.

        Primero intenta utilizar los documentos que
        BERTopic considera representativos.

        Si no están disponibles, utiliza documentos
        pertenecientes al topic como fallback.
        """
        try:
            representative_docs = model.get_representative_docs(topic_id)
        except Exception:
            representative_docs = []

        result = []

        if representative_docs:
            for text in representative_docs[:limit]:
                try:
                    index = documents.index(text)
                except ValueError:
                    continue

                result.append(
                    {
                        "document_id": document_ids[index],
                        "text": text,
                    }
                )

        if not result:
            for index in topic_indexes[:limit]:
                result.append(
                    {
                        "document_id": document_ids[index],
                        "text": documents[index],
                    }
                )

        return result