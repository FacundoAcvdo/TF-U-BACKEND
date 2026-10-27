import asyncio
import json
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any
import math
from chroma_repository import ChromaRepository
from config import get_settings
from embeddings import EmbeddingProvider
from models import (
    DatasetDocument,
    DatasetInfo,
    ChromaDocuments,
    Document,
)
from anonymization_service import AnonymizationService
from parsers import ParserFactory
from supabase_repository import (

    SupabaseRepository,
)
import logging
from logging_utils import setup_service_logger
logger = setup_service_logger("dataset-service")


def _format_chat_line(document: Document) -> str:
    """Prefija un mensaje como '[fecha] autor: texto'.

    El prefijo queda dentro del texto embebido y recuperado, para que
    el orden, el autor y la fecha sean útiles en la búsqueda y en el LLM.
    """
    fecha = document.fecha or "s/f"
    autor = (document.autor or "").strip() or "desconocido"
    return f"[{fecha}] {autor}: {document.texto}"


class DatasetService:

    def __init__(
        self,
        supabase_repository: SupabaseRepository,
        chroma_repository: ChromaRepository,
        embedding_provider: EmbeddingProvider,
        anonymization_service: AnonymizationService | None = None,
    ) -> None:

        self.supabase = supabase_repository

        self.chroma = chroma_repository

        self.embedding_provider = embedding_provider

        self.anonymization_service = (
            anonymization_service or AnonymizationService()
        )

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
        formato_fecha: str | None = None,
        columna_autor: str | None = None,
        es_chat: bool = False,
        user_id: str,
        user_token: str,
    ) -> DatasetInfo:

        dataset_id = str(
            uuid.uuid4()
        )


        dataset_name = await asyncio.to_thread(
            self.supabase.get_available_dataset_name,
            user_id=user_id,
            filename=filename,
            user_token=user_token,
        )


        phase_started = time.perf_counter()
        documents = await asyncio.to_thread(
            self._parse_documents,
            content=content,
            filename=filename,
            columna_texto=columna_texto,
            columna_clase=columna_clase,
            columna_fecha=columna_fecha,
            columna_autor=columna_autor,
            formato_original=formato_original,
            formato_detectado=formato_detectado,
            formato_fecha=formato_fecha,
            es_chat=es_chat,
        )
        parse_ms = int((time.perf_counter() - phase_started) * 1000)

        phase_started = time.perf_counter()
        anonymization = await asyncio.to_thread(
            self.anonymization_service.anonymize_documents,
            documents,
            es_chat=es_chat,
        )
        anonymization_ms = int((time.perf_counter() - phase_started) * 1000)

        documents = anonymization.documents

        phase_started = time.perf_counter()
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
        prepare_chroma_ms = int((time.perf_counter() - phase_started) * 1000)

        chunk_size = get_settings().ingest_chunk_size
        all_embeddings: list[list[float]] = []
        phase_metrics: dict[str, int] = {}

        try:
            phase_started = time.perf_counter()
            await self._embed_and_insert(
                ids=chroma_data.ids,
                documents=chroma_data.documents,
                metadatas=chroma_data.metadatas,
                chunk_size=chunk_size,
                all_embeddings=all_embeddings,
                collection_name=None,
                metrics=phase_metrics,
                metrics_prefix="documents",
            )
            phase_metrics["documents_total_ms"] = int(
                (time.perf_counter() - phase_started) * 1000
            )

            chunked_count = 0
            chunked_payload_bytes = 0

            if es_chat:
                phase_started = time.perf_counter()
                chunked = self._build_chat_chunks(
                    dataset_id=dataset_id,
                    user_id=user_id,
                    dataset_name=dataset_name,
                    documents=documents,
                )
                phase_metrics["chat_chunk_build_ms"] = int(
                    (time.perf_counter() - phase_started) * 1000
                )

                await self._embed_and_insert(
                    ids=chunked.ids,
                    documents=chunked.documents,
                    metadatas=chunked.metadatas,
                    chunk_size=chunk_size,
                    all_embeddings=None,
                    collection_name=(
                        get_settings().chroma_chunked_collection_name
                    ),
                    metrics=phase_metrics,
                    metrics_prefix="chunks",
                )
                phase_metrics["chunks_total_ms"] = int(
                    (time.perf_counter() - phase_started) * 1000
                )

                chunked_count = len(chunked.ids)

                chunked_payload_bytes = sum(
                    len(text.encode("utf-8"))
                    + len(json.dumps(meta, ensure_ascii=False).encode("utf-8"))
                    for text, meta in zip(chunked.documents, chunked.metadatas)
                )

            embedding_size_bytes = self.embedding_provider.calculate_embedding_size_bytes(
                all_embeddings
            )


            documents_size_bytes = sum(
                len(text.encode("utf-8"))
                + len(json.dumps(meta, ensure_ascii=False).encode("utf-8"))
                for text, meta in zip(chroma_data.documents, chroma_data.metadatas)
            )


            dimension = len(all_embeddings[0]) if all_embeddings else 0
            embedding_chunked_size_bytes = chunked_payload_bytes + chunked_count * dimension * 4

            await self.supabase.create_dataset(
                dataset_id=dataset.dataset_id,
                user_id=dataset.user_id,
                filename=dataset.filename,
                user_token=user_token,

                dataset_size=documents_size_bytes,
                embedding_size=embedding_size_bytes,
                embedding_chunked_size=embedding_chunked_size_bytes,
                tipo="chat" if es_chat else "dataset",
            )

            logger.info(
                "[dataset] ingest timings "
                f"parse_ms={parse_ms} "
                f"anonymization_ms={anonymization_ms} "
                f"prepare_chroma_ms={prepare_chroma_ms} "
                f"documents_embed_ms={phase_metrics.get('documents_embed_ms', 0)} "
                f"documents_insert_ms={phase_metrics.get('documents_insert_ms', 0)} "
                f"documents_total_ms={phase_metrics.get('documents_total_ms', 0)} "
                f"chat_chunk_build_ms={phase_metrics.get('chat_chunk_build_ms', 0)} "
                f"chunks_embed_ms={phase_metrics.get('chunks_embed_ms', 0)} "
                f"chunks_insert_ms={phase_metrics.get('chunks_insert_ms', 0)} "
                f"chunks_total_ms={phase_metrics.get('chunks_total_ms', 0)} "
                f"documents={len(documents)} "
                f"senders={anonymization.stats.remitentes} "
                f"anon_collect_ms={anonymization.stats.collect_senders_ms} "
                f"anon_regex_ms={anonymization.stats.build_sender_regex_ms} "
                f"anon_process_ms={anonymization.stats.process_documents_ms} "
                f"chunks={chunked_count}"
            )

        except Exception:

            await self._rollback_dataset(
                dataset=dataset,
                chroma_ids=chroma_data.ids,
                user_token=user_token,
            )

            raise

        return DatasetInfo(
            dataset_id=dataset.dataset_id,
            user_id=dataset.user_id,
            filename=dataset.filename,
            total_documents=len(
                dataset.documents
            ),
            created_at=dataset.created_at,
            anonymization=asdict(anonymization.stats),
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
        formato_fecha: str | None = None,
        columna_autor: str | None = None,
        es_chat: bool = False,
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


        columna_autor_resuelto = columna_autor
        if (
            es_chat
            and formato_detectado == "whatsapp-chat"
            and not columna_autor_resuelto
        ):
            columna_autor_resuelto = "remitente"

        documents = parser.parse(
            content=content,
            columna_texto=columna_texto,
            columna_clase=columna_clase,
            columna_fecha=columna_fecha,
            columna_autor=columna_autor_resuelto,
            extra_metadata=extra_metadata,
            formato_fecha=formato_fecha,
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
        metadatas: list[dict[str, Any]] = []

        for indice, document in enumerate(documents):
            document_id = str(uuid.uuid4())
            raw_meta = document.to_chroma_metadata()

            clean_meta: dict[str, str | int | float | bool] = {
                "dataset_id": str(dataset_id),
                "user_id": str(user_id),
                "source_filename": str(dataset_name or ""),


                "embedding_model": str(
                    getattr(
                        self.embedding_provider,
                        "model_name",
                        "unknown",
                    )
                ),
            }

            for k, v in raw_meta.items():
                if v is None:
                    continue
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    continue
                if isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                else:
                    clean_meta[k] = str(v)

            clean_meta["id"] = indice

            ids.append(document_id)
            texts.append(document.texto)
            metadatas.append(clean_meta)

        return ChromaDocuments(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
        )

    async def _embed_and_insert(
        self,
        *,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        chunk_size: int,
        all_embeddings: list[list[float]] | None,
        collection_name: str | None,
        metrics: dict[str, int] | None = None,
        metrics_prefix: str = "documents",
    ) -> None:
        """Embebe e inserta por bloques en threads (no bloquea el loop)."""
        total = len(ids)

        for inicio in range(0, total, chunk_size):
            fin = min(inicio + chunk_size, total)

            embed_started = time.perf_counter()
            embeddings = await asyncio.to_thread(
                self.embedding_provider.embed_documents,
                documents[inicio:fin],
            )
            if metrics is not None:
                metrics[f"{metrics_prefix}_embed_ms"] = (
                    metrics.get(f"{metrics_prefix}_embed_ms", 0)
                    + int((time.perf_counter() - embed_started) * 1000)
                )

            if all_embeddings is not None:
                all_embeddings.extend(embeddings)

            insert_started = time.perf_counter()
            await asyncio.to_thread(
                self.chroma.add_documents,
                ids=ids[inicio:fin],
                documents=documents[inicio:fin],
                metadatas=metadatas[inicio:fin],
                embeddings=embeddings,
                collection_name=collection_name,
            )
            if metrics is not None:
                metrics[f"{metrics_prefix}_insert_ms"] = (
                    metrics.get(f"{metrics_prefix}_insert_ms", 0)
                    + int((time.perf_counter() - insert_started) * 1000)
                )

    def _build_chat_chunks(
        self,
        *,
        dataset_id: str,
        user_id: str,
        dataset_name: str,
        documents: list[Document],
    ) -> ChromaDocuments:
        """Ventanas de N mensajes con solapamiento, ordenadas por fecha.

        Cada fila es un mensaje. Se ordena por fecha (sin fecha al final,
        orden estable) y se emiten ventanas de `chat_chunk_size` mensajes
        con `chat_chunk_overlap` de solapamiento.
        """
        settings = get_settings()
        size = max(1, settings.chat_chunk_size)
        overlap = max(0, min(settings.chat_chunk_overlap, size - 1))
        paso = size - overlap

        ordenados = sorted(
            documents,
            key=lambda d: (d.fecha is None, d.fecha or ""),
        )

        total = len(ordenados)
        ids: list[str] = []
        texts: list[str] = []
        metadatas: list[dict[str, Any]] = []

        embedding_model = str(
            getattr(self.embedding_provider, "model_name", "unknown")
        )

        chunk_index = 0
        inicio = 0
        while inicio < total:
            fin = min(inicio + size, total)
            ventana = ordenados[inicio:fin]

            fechas = [d.fecha for d in ventana if d.fecha]

            meta: dict[str, str | int | float | bool] = {
                "dataset_id": str(dataset_id),
                "user_id": str(user_id),
                "source_filename": str(dataset_name or ""),
                "embedding_model": embedding_model,
                "chunk_index": chunk_index,
                "chunk_size": len(ventana),
            }

            if fechas:
                meta["fecha_desde"] = fechas[0]
                meta["fecha_hasta"] = fechas[-1]

            ids.append(str(uuid.uuid4()))
            texts.append(
                "\n".join(
                    _format_chat_line(d) for d in ventana
                )
            )
            metadatas.append(meta)

            chunk_index += 1
            inicio += paso

        return ChromaDocuments(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
        )

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

            self.chroma.delete_chunked_by_dataset(
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
            logger.exception(f"Error retrieving datasets for user {user_id}: {e}")
            raise

    async def delete_dataset(
        self,
        *,
        dataset_id: str,
        user_id: str,
        user_token: str,
    ) -> None:
        """Borra solo el registro (rápido). Los vectores van en background."""

        if not dataset_id:
            raise ValueError(
                "El dataset_id no puede estar vacío."
            )

        try:

            await self.supabase.delete_dataset(
                dataset_id=dataset_id,
                user_id=user_id,
                user_token=user_token,
            )

        except Exception as e:
            logger.exception(f"Error deleting dataset {dataset_id} for user {user_id}: {e}")
            raise

    def delete_dataset_vectors(
        self,
        *,
        dataset_id: str,
        attempts: int = 3,
    ) -> None:
        """Borra vectores en background con verificación y reintentos."""

        where = {"dataset_id": dataset_id}

        for attempt in range(1, attempts + 1):
            try:
                self.chroma.delete_documents_by_dataset(
                    dataset_id=dataset_id,
                )
                self.chroma.delete_chunked_by_dataset(
                    dataset_id=dataset_id,
                )

                if not self.chroma.has_remaining(where):
                    pass
                    return

                pass

            except Exception as e:
                logger.exception(f"[dataset] Borrado de vectores {dataset_id} "

                      f"intento {attempt}/{attempts} falló: {e}")

            time.sleep(5 * attempt)

        logger.warning(f"[dataset] WARNING: quedaron vectores huérfanos "

              f"del dataset {dataset_id}.")

    async def delete_datasets_by_user(
        self,
        *,
        user_id: str,
        user_token: str,
    ) -> None:
        """Borra solo los registros (rápido). Los vectores van en background."""

        try:

            await self.supabase.delete_datasets_by_user(
                user_id=user_id,
                user_token=user_token,
            )

        except Exception as e:
            logger.exception(f"Error deleting datasets for user {user_id}: {e}")
            raise

    def delete_user_vectors(
        self,
        *,
        user_id: str,
        attempts: int = 3,
    ) -> None:
        """Borra vectores en background con verificación y reintentos."""

        where = {"user_id": user_id}

        for attempt in range(1, attempts + 1):
            try:
                self.chroma.delete_documents_by_user(
                    user_id=user_id,
                )
                self.chroma.delete_chunked_by_user(
                    user_id=user_id,
                )

                if not self.chroma.has_remaining(where):
                    pass
                    return

                pass

            except Exception as e:
                logger.exception(f"[dataset] Borrado de vectores {user_id} "

                      f"intento {attempt}/{attempts} falló: {e}")

            time.sleep(5 * attempt)

        logger.warning(f"[dataset] WARNING: quedaron vectores huérfanos "

              f"del usuario {user_id}.")
