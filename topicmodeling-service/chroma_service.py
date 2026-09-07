import chromadb

from config import get_settings


class ChromaService:

    def __init__(self) -> None:

        settings = get_settings()

        self.client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
        )

    async def get_dataset_documents(
        self,
        *,
        dataset_id: str,
        user_id: str,
    ):

        collection = (
            self.client.get_collection(
                name = get_settings().chroma_collection_name
            )
        )

        data = collection.get(
            where={
                "$and": [
                    {"dataset_id": dataset_id},
                    {"user_id": user_id},
                ]
            },
            include=[
                "documents",
                "embeddings",
                "metadatas",
            ],
        )

        return {
            "ids": data.get("ids", []),
            "documents": data.get(
                "documents",
                [],
            ),
            "embeddings": data.get(
                "embeddings",
                [],
            ),
            "metadatas": data.get(
                "metadatas",
                [],
            ),
        }

