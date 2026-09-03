import asyncio
from typing import Any

from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
import nltk
from nltk.corpus import stopwords
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


class TopicModelingProvider:

    def __init__(self) -> None:
        self.model_config = {
            "verbose": False,
        }

        # Descarga silenciosa del corpus de stopwords si no está presente
        nltk.download("stopwords", quiet=True)

        # Unificación de stopwords de inglés y español
        raw_stopwords = set(
            stopwords.words("english") + stopwords.words("spanish")
        )

        # Palabras críticas en salud mental/suicidio que NO deben filtrarse
        critical_words = {
            "no",
            "not",
            "never",
            "nunca",
            "jamás",
            "jamas",
            "nada",
            "nadie",
            "sin",
            "without",
            "alone",
            "solo",
            "sola",
            "solos",
            "solas",
        }

        self.multilingual_stopwords = list(raw_stopwords - critical_words)

    async def extract_topics(
        self,
        *,
        documents: list[str],
        embeddings: list[list[float]],
        document_ids: list[str],
    ) -> dict[str, Any]:

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

        print("Iniciando topic modeling con BERTopic")

        if documents is None or len(documents) == 0:
            raise ValueError("No hay documentos para realizar topic modeling.")

        if embeddings is None or len(embeddings) == 0:
            raise ValueError("No hay embeddings para realizar topic modeling.")

        if len(documents) != len(embeddings):
            raise ValueError(
                "La cantidad de documentos y embeddings no coincide."
            )

        if len(documents) != len(document_ids):
            raise ValueError("La cantidad de documentos e IDs no coincide.")

        embeddings_np = np.asarray(embeddings, dtype=np.float32)

        # 1. Extractor con soporte bilingüe y n-gramas
        vectorizer_model = CountVectorizer(
            stop_words=self.multilingual_stopwords,
            ngram_range=(1, 3),
            min_df=2,
        )

        # 2. c-TF-IDF con penalización de términos omnipresentes
        ctfidf_model = ClassTfidfTransformer(
            reduce_frequent_words=True,
            bm25_weighting=True,
        )

        print("Iniciando topic modeling con BERTopic")

        # 3. BERTopic utilizando los embeddings existentes de Chroma
        model = BERTopic(
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf_model,
            min_topic_size=10,
            **self.model_config,
        )

        print("Ejecutando topic modeling con BERTopic")

        topics_raw, probabilities = model.fit_transform(
            documents,
            embeddings=embeddings_np,
        )

        topics = [int(t) for t in topics_raw]

        print("reduciendo outliers")

        # 4. Reducción de outliers reasignándolos al clúster más cercano
        try:
            new_topics = model.reduce_outliers(
                documents,
                topics,
                strategy="c-tf-idf",
                embeddings=embeddings_np,
            )
            model.update_topics(
                documents,
                topics=new_topics,
                vectorizer_model=vectorizer_model,
                ctfidf_model=ctfidf_model,
            )
            topics = [int(t) for t in new_topics]
        except Exception:
            # Fallback en caso de que no haya suficientes documentos por clúster
            pass

        document_count = len(documents)
        outlier_count = topics.count(-1)
        outlier_percentage = (
            (outlier_count / document_count) * 100 if document_count > 0 else 0
        )

        unique_topics = sorted(set(topics))
        valid_topics = [
            topic_id for topic_id in unique_topics if topic_id != -1
        ]

        topic_results = []

        for topic_id in valid_topics:
            topic_indexes = [
                index
                for index, assigned_topic in enumerate(topics)
                if assigned_topic == topic_id
            ]

            topic_document_count = len(topic_indexes)
            percentage = (topic_document_count / document_count) * 100

            topic_words = model.get_topic(topic_id)
            keywords = (
                [word for word, score in topic_words[:10]]
                if topic_words
                else []
            )

            representative_documents = self._get_representative_documents(
                model=model,
                topic_id=topic_id,
                documents=documents,
                document_ids=document_ids,
                topic_indexes=topic_indexes,
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