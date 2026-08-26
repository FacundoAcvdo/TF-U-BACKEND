from abc import ABC, abstractmethod

class EmbeddingProvider(ABC):

    @abstractmethod
    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        raise NotImplementedError

class NoEmbeddingProvider(
    EmbeddingProvider
):

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        raise NotImplementedError(
            "No hay un modelo de embeddings configurado."
        )