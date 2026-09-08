from abc import ABC, abstractmethod
import torch
from sentence_transformers import SentenceTransformer


class EmbeddingProvider(ABC):

    @abstractmethod
    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        raise NotImplementedError

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

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
    
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=self.device)

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        if not texts:
            return []

        embeddings = self.model.encode(
            sentences=texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embeddings.tolist()