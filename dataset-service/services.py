import asyncio
import json
import time
import uuid
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
from parsers import ParserFactory
from supabase_repository import (
    SupabaseRepository,
)


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
    ) -> None:

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
        formato_fecha: str | None = None,
        columna_autor: str | None = None,
        es_chat: bool = False,
        user_id: str,
        user_token: str,
    ) -> DatasetInfo:

        dataset_id = str(
            uuid.uuid4()
        )

        # Lookup de nombre en thread: supabase-py es bloqueante.
        dataset_name = await asyncio.to_thread(
            self.supabase.get_available_dataset_name,
            user_id=user_id,
            filename=filename,
            user_token=user_token,
        )

        # Parseo en thread: CPU-bound (pandas/polars).
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

        # Embed + insert por chunks (streaming): acota RAM y solapa GPU/red.
        # Sin materializar N vectores antes del primer insert.
        chunk_size = get_settings().ingest_chunk_size
        all_embeddings: list[list[float]] = []

        try:
            await self._embed_and_insert(
                ids=chroma_data.ids,
                documents=chroma_data.documents,
                metadatas=chroma_data.metadatas,
                chunk_size=chunk_size,
                all_embeddings=all_embeddings,
                collection_name=None,
            )

            chunked_count = 0
            chunked_payload_bytes = 0

            if es_chat:
                chunked = self._build_chat_chunks(
                    dataset_id=dataset_id,
                    user_id=user_id,
                    dataset_name=dataset_name,
                    documents=documents,
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

            # Payload real en Chroma (documentos + metadata), no el archivo crudo.
            documents_size_bytes = sum(
                len(text.encode("utf-8"))
                + len(json.dumps(meta, ensure_ascii=False).encode("utf-8"))
                for text, meta in zip(chroma_data.documents, chroma_data.metadatas)
            )

            # Vectores de chunks: misma dimensión que los principales.
            dimension = len(all_embeddings[0]) if all_embeddings else 0
            embedding_chunked_size_bytes = chunked_payload_bytes + chunked_count * dimension * 4

            await self.supabase.create_dataset(
                dataset_id=dataset.dataset_id,
                user_id=dataset.user_id,
                filename=dataset.filename,
                user_token=user_token,
                # Documentos + metadata tal como quedan en Chroma.
                dataset_size=documents_size_bytes,
                embedding_size=embedding_size_bytes,
                embedding_chunked_size=embedding_chunked_size_bytes,
                tipo="chat" if es_chat else "dataset",
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

        # Chats de WhatsApp: el .txt se convierte a CSV con columnas
        # fijas fecha,hora,remitente,mensaje, por lo que el autor se infiere
        # sin pedirlo en el front.
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
                # Versionado: permite detectar vectores de otro modelo
                # (las dims deben coincidir dentro de la colección).
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
    ) -> None:
        """Embebe e inserta por bloques en threads (no bloquea el loop)."""
        total = len(ids)

        for inicio in range(0, total, chunk_size):
            fin = min(inicio + chunk_size, total)

            embeddings = await asyncio.to_thread(
                self.embedding_provider.embed_documents,
                documents[inicio:fin],
            )

            if all_embeddings is not None:
                all_embeddings.extend(embeddings)

            await asyncio.to_thread(
                self.chroma.add_documents,
                ids=ids[inicio:fin],
                documents=documents[inicio:fin],
                metadatas=metadatas[inicio:fin],
                embeddings=embeddings,
                collection_name=collection_name,
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
            print(f"Error retrieving datasets for user {user_id}: {e}")
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
            print(f"Error deleting dataset {dataset_id} for user {user_id}: {e}")
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
                    print(f"[dataset] Vectores del dataset {dataset_id} eliminados.", flush=True)
                    return

                print(f"[dataset] Aún quedan vectores del dataset "
                      f"{dataset_id}, reintentando ({attempt}/{attempts})...", flush=True)

            except Exception as e:
                print(f"[dataset] Borrado de vectores {dataset_id} "
                      f"intento {attempt}/{attempts} falló: {e}", flush=True)

            time.sleep(5 * attempt)

        print(f"[dataset] WARNING: quedaron vectores huérfanos "
              f"del dataset {dataset_id}.", flush=True)

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
            print(f"Error deleting datasets for user {user_id}: {e}")
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
                    print(f"[dataset] Vectores del usuario {user_id} eliminados.", flush=True)
                    return

                print(f"[dataset] Aún quedan vectores del usuario "
                      f"{user_id}, reintentando ({attempt}/{attempts})...", flush=True)

            except Exception as e:
                print(f"[dataset] Borrado de vectores {user_id} "
                      f"intento {attempt}/{attempts} falló: {e}", flush=True)

            time.sleep(5 * attempt)

        print(f"[dataset] WARNING: quedaron vectores huérfanos "
              f"del usuario {user_id}.", flush=True)
