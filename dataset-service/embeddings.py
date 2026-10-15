from abc import ABC, abstractmethod
import os
import threading

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

import model_cache
from model_cache import ModelSlot


DEFAULT_MODEL = os.environ.get(
    "EMBEDDING_MODEL",
    "intfloat/multilingual-e5-large",
)
DEFAULT_BATCH_SIZE = int(os.environ.get("EMBEDDING_BATCH_SIZE", "64"))
DEFAULT_DEVICE = os.environ.get("EMBEDDING_DEVICE", "auto")
DEFAULT_FP16 = os.environ.get("EMBEDDING_FP16", "true").lower() in (
    "true", "1", "yes", "on",
)
EMBEDDING_VRAM_MB = int(os.environ.get("EMBEDDING_VRAM_MB", "1500"))


class EmbeddingProvider(ABC):

    model_name: str = "unknown"

    @abstractmethod
    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        raise NotImplementedError(
            "Este provider no implementa embed_query."
        )

    def calculate_embedding_size_bytes(
        self,
        embeddings: list[list[float]],
    ) -> int:
        if not embeddings:
            return 0

        dimension = len(embeddings[0])
        bytes_per_float32 = 4
        return len(embeddings) * dimension * bytes_per_float32


class NoEmbeddingProvider(EmbeddingProvider):

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        raise NotImplementedError(
            "No hay un modelo de embeddings configurado."
        )


class LocalSentenceTransformerProvider(EmbeddingProvider):
    """Embeddings E5 (por defecto multilingual-e5-large, 1024 dims).

    E5 exige prefijos: "passage: " para documentos indexados y
    "query: " para consultas. Sin ellos la calidad cae fuerte.
    """

    passage_prefix: str = "passage: "
    query_prefix: str = "query: "

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: str | None = None,
        fp16: bool = DEFAULT_FP16,
    ) -> None:

        if device is None or device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_name = model_name
        self.batch_size = batch_size
        self._fp16 = fp16
        self._slot = None
        self._slot_lock = threading.Lock()

    def _get_slot(self):
        slot = self._slot
        if slot is None:
            with self._slot_lock:
                slot = self._slot
                if slot is None:
                    provider = self

                    def _load():
                        model_kwargs: dict = {}
                        if provider._fp16 and provider.device == "cuda":
                            model_kwargs["torch_dtype"] = torch.float16
                        return SentenceTransformer(
                            provider.model_name,
                            device=provider.device,
                            model_kwargs=model_kwargs or None,
                        )

                    def _drop(model):
                        try:
                            del model
                        except Exception:
                            pass

                    slot = ModelSlot(
                        name=f"embeddings-{self.model_name}",
                        vram_mb=EMBEDDING_VRAM_MB,
                        load_fn=_load,
                        unload_fn=_drop,
                    )
                    self._slot = slot
        return slot

    def _with_prefix(self, texts: list[str], prefix: str) -> list[str]:
        return [
            f"{prefix}{t}" if not t.startswith(prefix) else t
            for t in texts
        ]

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        if not texts:
            return []

        slot = self._get_slot()
        slot.maybe_evict()
        with slot.use() as model:
            embeddings = model.encode(
                sentences=self._with_prefix(texts, self.passage_prefix),
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

        return np.asarray(embeddings, dtype=np.float32).tolist()

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        slot = self._get_slot()
        slot.maybe_evict()
        with slot.use() as model:
            embeddings = model.encode(
                sentences=[f"{self.query_prefix}{text}"],
                batch_size=1,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

        return np.asarray(embeddings, dtype=np.float32)[0].tolist()
