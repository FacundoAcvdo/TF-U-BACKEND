import asyncio
from collections import defaultdict
from typing import Any

import hdbscan
import nltk
import numpy as np
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from nltk.corpus import stopwords
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP


class TopicModelingProvider:
    def __init__(
        self,
        min_topic_size_ratio: float = 0.001,
        min_topic_size: int = 10,
        large_dataset_threshold: int = 40_000,
    ) -> None:

        if min_topic_size_ratio <= 0:
            raise ValueError("min_topic_size_ratio debe ser mayor que 0.")

        if min_topic_size < 2:
            raise ValueError("min_topic_size debe ser mayor o igual a 2.")

        self.min_topic_size_ratio = min_topic_size_ratio
        self.min_topic_size = min_topic_size
        self.large_dataset_threshold = large_dataset_threshold

        self.model_config = {
            "verbose": False,
        }

        self._download_stopwords()

    # -------------------------------------------------------------------------
    # Inicialización
    # -------------------------------------------------------------------------

    @staticmethod
    def _download_stopwords() -> None:
        nltk.download("stopwords", quiet=True)

    def _build_stopwords(self) -> list[str]:
        raw_stopwords = set(
            stopwords.words("english")
            + stopwords.words("spanish")
        )

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

        return list(raw_stopwords - critical_words)

    # -------------------------------------------------------------------------
    # API pública
    # -------------------------------------------------------------------------

    async def extract_topics(
        self,
        *,
        documents: list[str],
        embeddings: list[list[float]],
        document_ids: list[str],
        max_topics: int=30,
    ) -> dict[str, Any]:

        return await asyncio.to_thread(
            self._extract_topics_sync,
            documents,
            embeddings,
            document_ids,
            max_topics,
        )

    # -------------------------------------------------------------------------
    # Pipeline principal
    # -------------------------------------------------------------------------

    def _extract_topics_sync(
        self,
        documents: list[str],
        embeddings: list[list[float]],
        document_ids: list[str],
        max_topics: int,
    ) -> dict[str, Any]:

        self._validate_input(
            documents=documents,
            embeddings=embeddings,
            document_ids=document_ids,
        )

        n_docs = len(documents)

        embeddings_np = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        if embeddings_np.ndim != 2:
            raise ValueError(
                "Los embeddings deben ser un array 2D."
            )

        # ------------------------------------------------------------------
        # 1. Reducción dimensional
        # ------------------------------------------------------------------

        dim_reduction_model = self._build_dimension_reduction_model(
            n_docs=n_docs,
        )

        # ------------------------------------------------------------------
        # 2. HDBSCAN
        # ------------------------------------------------------------------

        hdbscan_model = self._build_hdbscan_model(
            n_docs=n_docs,
        )

        # ------------------------------------------------------------------
        # 3. Vectorizer
        # ------------------------------------------------------------------

        vectorizer_model = self._build_vectorizer(
            n_docs=n_docs,
        )

        # ------------------------------------------------------------------
        # 4. c-TF-IDF
        # ------------------------------------------------------------------

        ctfidf_model = ClassTfidfTransformer(
            reduce_frequent_words=True,
            bm25_weighting=True,
        )

        # ------------------------------------------------------------------
        # 5. BERTopic
        # ------------------------------------------------------------------

        model = BERTopic(
            umap_model=dim_reduction_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf_model,
            **self.model_config,
        )

        topics_raw, _ = model.fit_transform(
            documents,
            embeddings=embeddings_np,
        )

        topics = np.asarray(
            topics_raw,
            dtype=np.int32,
        )

        # ------------------------------------------------------------------
        # 6. Reducir outliers
        # ------------------------------------------------------------------

        topics = self._reduce_outliers(
            model=model,
            documents=documents,
            topics=topics,
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf_model,
        )

        # ------------------------------------------------------------------
        # 7. Reducir cantidad de tópicos SOLO si supera max_topics
        # ------------------------------------------------------------------

        topics = self._reduce_topics_if_needed(
            model=model,
            documents=documents,
            topics=topics,
            max_topics=max_topics,
        )

        # ------------------------------------------------------------------
        # 8. Construcción eficiente de índices por tópico
        # ------------------------------------------------------------------

        topic_indexes = self._build_topic_indexes(
            topics=topics,
        )

        # ------------------------------------------------------------------
        # 9. Construcción de resultados
        # ------------------------------------------------------------------

        topic_results = self._build_topic_results(
            model=model,
            documents=documents,
            document_ids=document_ids,
            topics=topics,
            topic_indexes=topic_indexes,
        )

        topic_results.sort(
            key=lambda topic: topic["document_count"],
            reverse=True,
        )

        outlier_count = int(np.count_nonzero(topics == -1))

        return {
            "document_count": n_docs,
            "topic_count": len(topic_results),
            "outlier_count": outlier_count,
            "outlier_percentage": round(
                (outlier_count / n_docs) * 100,
                2,
            ),
            "topics": topic_results,
        }

    # -------------------------------------------------------------------------
    # Validación
    # -------------------------------------------------------------------------

    @staticmethod
    def _validate_input(
        *,
        documents: list[str],
        embeddings: list[list[float]],
        document_ids: list[str],
    ) -> None:

        if documents is None or len(documents) == 0:
            raise ValueError(
                "No hay documentos para realizar topic modeling."
            )

        if embeddings is None or len(embeddings) == 0:
            raise ValueError(
                "No hay embeddings para realizar topic modeling."
            )

        if len(documents) != len(embeddings):
            raise ValueError(
                "La cantidad de documentos y embeddings no coincide."
            )

        if document_ids is None or len(document_ids) == 0:
            raise ValueError(
                "No hay document_ids."
            )

        if len(documents) != len(document_ids):
            raise ValueError(
                "La cantidad de documentos e IDs no coincide."
            )

    # -------------------------------------------------------------------------
    # UMAP / PCA
    # -------------------------------------------------------------------------

    def _build_dimension_reduction_model(
        self,
        *,
        n_docs: int,
    ):

        if n_docs > self.large_dataset_threshold:
            return PCA(
                n_components=5,
                random_state=42,
            )

        return UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric="cosine",
            random_state=42,
            n_jobs=-1,
            low_memory=True,
        )

    # -------------------------------------------------------------------------
    # HDBSCAN
    # -------------------------------------------------------------------------

    def _build_hdbscan_model(
        self,
        *,
        n_docs: int,
    ) -> hdbscan.HDBSCAN:

        min_cluster_size = max(
            self.min_topic_size,
            int(n_docs * self.min_topic_size_ratio),
        )

        return hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
            core_dist_n_jobs=-1,
        )

    # -------------------------------------------------------------------------
    # Vectorizer
    # -------------------------------------------------------------------------

    def _build_vectorizer(
        self,
        *,
        n_docs: int,
    ) -> CountVectorizer:

        min_df = 5 if n_docs > 5_000 else 2

        return CountVectorizer(
            stop_words=self._build_stopwords(),
            ngram_range=(1, 2),
            min_df=min_df,
        )

    # -------------------------------------------------------------------------
    # Outliers
    # -------------------------------------------------------------------------

    def _reduce_outliers(
        self,
        *,
        model: BERTopic,
        documents: list[str],
        topics: np.ndarray,
        vectorizer_model: CountVectorizer,
        ctfidf_model: ClassTfidfTransformer,
    ) -> np.ndarray:

        if not np.any(topics == -1):
            return topics

        try:
            new_topics = model.reduce_outliers(
                documents,
                topics.tolist(),
                strategy="c-tf-idf",
            )

            new_topics = np.asarray(
                new_topics,
                dtype=np.int32,
            )

            model.update_topics(
                documents,
                topics=new_topics.tolist(),
                vectorizer_model=vectorizer_model,
                ctfidf_model=ctfidf_model,
            )

            return new_topics

        except Exception:
            return topics

    # -------------------------------------------------------------------------
    # Reducción de tópicos
    # -------------------------------------------------------------------------

    def _reduce_topics_if_needed(
        self,
        *,
        model: BERTopic,
        documents: list[str],
        topics: np.ndarray,
        max_topics: int,
    ) -> np.ndarray:

        unique_topics = np.unique(topics)

        current_topic_count = int(
            np.count_nonzero(unique_topics != -1)
        )

        if current_topic_count <= max_topics:
            return topics

        try:
            model.reduce_topics(
                documents,
                nr_topics=max_topics,
            )

            reduced_topics = np.asarray(
                model.topics_,
                dtype=np.int32,
            )

            return reduced_topics

        except Exception as e:
            print(f"[ERROR] No fue posible reducir los tópicos de "
                f"{current_topic_count} a {str(max_topics)}. Se mantiene la cantidad original de tópicos.")
            raise RuntimeError(
                f"No fue posible reducir los tópicos de "
                f"{current_topic_count} a {str(max_topics)}."
                f" Error: {str(e)}"
            )

    # -------------------------------------------------------------------------
    # Índices por tópico
    # -------------------------------------------------------------------------

    @staticmethod
    def _build_topic_indexes(
        *,
        topics: np.ndarray,
    ) -> dict[int, list[int]]:

        topic_indexes: dict[int, list[int]] = defaultdict(list)

        for index, topic_id in enumerate(topics):
            topic_id = int(topic_id)

            if topic_id != -1:
                topic_indexes[topic_id].append(index)

        return dict(topic_indexes)

    # -------------------------------------------------------------------------
    # Resultados
    # -------------------------------------------------------------------------

    def _build_topic_results(
        self,
        *,
        model: BERTopic,
        documents: list[str],
        document_ids: list[str],
        topics: np.ndarray,
        topic_indexes: dict[int, list[int]],
    ) -> list[dict[str, Any]]:

        n_docs = len(documents)

        doc_to_indexes: dict[str, list[int]] = defaultdict(list)

        for index, document in enumerate(documents):
            doc_to_indexes[document].append(index)

        results: list[dict[str, Any]] = []

        for topic_id, indexes in topic_indexes.items():

            topic_words = model.get_topic(topic_id) or []

            keywords = [
                word
                for word, _ in topic_words[:10]
            ]

            representative_documents = (
                self._get_representative_documents(
                    model=model,
                    topic_id=topic_id,
                    documents=documents,
                    document_ids=document_ids,
                    topic_indexes=indexes,
                    doc_to_indexes=doc_to_indexes,
                    limit=3,
                )
            )

            document_count = len(indexes)

            results.append({
                "topic_id": int(topic_id),
                "document_count": document_count,
                "percentage": round(
                    (document_count / n_docs) * 100,
                    2,
                ),
                "keywords": keywords,
                "representative_documents": representative_documents,
            })

        return results

    # -------------------------------------------------------------------------
    # Documentos representativos
    # -------------------------------------------------------------------------

    @staticmethod
    def _get_representative_documents(
        *,
        model: BERTopic,
        topic_id: int,
        documents: list[str],
        document_ids: list[str],
        topic_indexes: list[int],
        doc_to_indexes: dict[str, list[int]],
        limit: int = 3,
    ) -> list[dict[str, str]]:

        result: list[dict[str, str]] = []

        try:
            representative_documents = (
                model.get_representative_docs(topic_id) or []
            )
        except Exception:
            representative_documents = []

        used_indexes: set[int] = set()

        for text in representative_documents:

            indexes = doc_to_indexes.get(text)

            if not indexes:
                continue

            selected_index = None

            # Intentamos encontrar un índice todavía no utilizado.
            for index in indexes:
                if index not in used_indexes:
                    selected_index = index
                    break

            if selected_index is None:
                continue

            used_indexes.add(selected_index)

            result.append({
                "document_id": document_ids[selected_index],
                "text": text,
            })

            if len(result) >= limit:
                return result

        # Fallback.
        if len(result) < limit:

            for index in topic_indexes:

                if index in used_indexes:
                    continue

                result.append({
                    "document_id": document_ids[index],
                    "text": documents[index],
                })

                if len(result) >= limit:
                    break

        return result