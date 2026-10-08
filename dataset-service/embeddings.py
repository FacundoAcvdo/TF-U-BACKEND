from abc import ABC, abstractmethod
import os

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = os.environ.get(
    "EMBEDDING_MODEL",
    "intfloat/multilingual-e5-large",
)
DEFAULT_BATCH_SIZE = int(os.environ.get("EMBEDDING_BATCH_SIZE", "64"))
DEFAULT_DEVICE = os.environ.get("EMBEDDING_DEVICE", "auto")
DEFAULT_FP16 = os.environ.get("EMBEDDING_FP16", "true").lower() in (
    "true", "1", "yes", "on",
)


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

        model_kwargs: dict = {}
        if fp16 and self.device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16

        self.model = SentenceTransformer(
            model_name,
            device=self.device,
            model_kwargs=model_kwargs or None,
        )

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

        embeddings = self.model.encode(
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
        embeddings = self.model.encode(
            sentences=[f"{self.query_prefix}{text}"],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return np.asarray(embeddings, dtype=np.float32)[0].tolist()
