import asyncio
from typing import Any
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
import hdbscan
import nltk
from nltk.corpus import stopwords
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP


class TopicModelingProvider:

    def __init__(self) -> None:
        self.model_config = {"verbose": False}

        nltk.download("stopwords", quiet=True)
        raw_stopwords = set(
            stopwords.words("english") + stopwords.words("spanish")
        )
        critical_words = {
            "no", "not", "never", "nunca", "jamás", "jamas", "nada",
            "nadie", "sin", "without", "alone", "solo", "sola", "solos", "solas",
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
        if documents is None or len(documents) == 0:
            raise ValueError("No hay documentos para realizar topic modeling.")

        if embeddings is None or len(embeddings) == 0:
            raise ValueError("No hay embeddings para realizar topic modeling.")

        if len(documents) != len(embeddings):
            raise ValueError("La cantidad de documentos y embeddings no coincide.")

        if document_ids is None or len(documents) != len(document_ids):
            raise ValueError("La cantidad de documentos e IDs no coincide.")

        n_docs = len(documents)
        embeddings_np = np.asarray(embeddings, dtype=np.float32)

        # 1. Configuración adaptativa de reducción según volumen
        if n_docs > 40_000:
            # PCA es casi instantáneo en CPU para datasets masivos
            dim_reduction_model = PCA(n_components=5)
        else:
            # UMAP paralelizado al 100% de la CPU
            dim_reduction_model = UMAP(
                n_neighbors=15,
                n_components=5,
                min_dist=0.0,
                metric="cosine",
                random_state=42,
                n_jobs=-1,  # Usa todos los cores de CPU
                low_memory=True,
            )

        # 2. HDBSCAN con todos los núcleos asignados
        hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=max(10, int(n_docs * 0.001)),
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
            core_dist_n_jobs=-1,  # Paraleliza el cálculo de distancias
        )

        # 3. Vectorizer con poda de vocabulario para no saturar memoria
        vectorizer_model = CountVectorizer(
            stop_words=self.multilingual_stopwords,
            ngram_range=(1, 2),  # (1, 2) es 3x más rápido en CPU que (1, 3)
            min_df=5 if n_docs > 5_000 else 2,
        )

        ctfidf_model = ClassTfidfTransformer(
            reduce_frequent_words=True,
            bm25_weighting=True,
        )

        # 4. Inicialización e inferencia
        model = BERTopic(
            umap_model=dim_reduction_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf_model,
            **self.model_config,
        )

        topics_raw, _ = model.fit_transform(documents, embeddings=embeddings_np)
        topics = [int(t) for t in topics_raw]

        # 5. Reducción de outliers
        try:
            new_topics = model.reduce_outliers(
                documents,
                topics,
                strategy="c-tf-idf",
            )
            model.update_topics(
                documents,
                topics=new_topics,
                vectorizer_model=vectorizer_model,
                ctfidf_model=ctfidf_model,
            )
            topics = [int(t) for t in new_topics]
        except Exception:
            pass

        # 6. Post-procesamiento O(1) creando un mapa indexado previo
        doc_to_idx = {doc: i for i, doc in enumerate(documents)}

        unique_topics = [t for t in sorted(set(topics)) if t != -1]
        topic_results = []

        for topic_id in unique_topics:
            topic_indexes = [i for i, t in enumerate(topics) if t == topic_id]
            topic_words = model.get_topic(topic_id) or []
            keywords = [word for word, _ in topic_words[:10]]

            representative_documents = self._get_representative_documents_fast(
                model=model,
                topic_id=topic_id,
                documents=documents,
                document_ids=document_ids,
                topic_indexes=topic_indexes,
                doc_to_idx=doc_to_idx,
            )

            topic_results.append({
                "topic_id": topic_id,
                "document_count": len(topic_indexes),
                "percentage": round((len(topic_indexes) / n_docs) * 100, 2),
                "keywords": keywords,
                "representative_documents": representative_documents,
            })

        topic_results.sort(key=lambda t: t["document_count"], reverse=True)

        return {
            "document_count": n_docs,
            "topic_count": len(topic_results),
            "outlier_percentage": round((topics.count(-1) / n_docs) * 100, 2),
            "topics": topic_results,
        }

    def _get_representative_documents_fast(
        self,
        *,
        model: BERTopic,
        topic_id: int,
        documents: list[str],
        document_ids: list[str],
        topic_indexes: list[int],
        doc_to_idx: dict[str, int],
        limit: int = 3,
    ) -> list[dict[str, str]]:
        result = []
        try:
            repr_docs = model.get_representative_docs(topic_id) or []
        except Exception:
            repr_docs = []

        for text in repr_docs[:limit]:
            idx = doc_to_idx.get(text)
            if idx is not None:
                result.append({"document_id": document_ids[idx], "text": text})

        if not result:
            for idx in topic_indexes[:limit]:
                result.append({"document_id": document_ids[idx], "text": documents[idx]})

        return result