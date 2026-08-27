from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

@dataclass
class Document:

    texto: str

    clase: str | None = None

    fecha: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:

        return {
            "texto": self.texto,
            "clase": self.clase,
            "fecha": self.fecha,
            "metadata": self.metadata,
        }

    def to_chroma_metadata(
        self,
    ) -> dict[str, Any]:

        result: dict[str, Any] = {}

        if self.clase is not None:
            result["clase"] = str(self.clase)

        if self.fecha is not None:
            result["fecha"] = str(self.fecha)

        for key, value in self.metadata.items():

            if value is None:
                continue

            if isinstance(value, (str, int, float, bool)):
                result[key] = value
            else:
                result[key] = str(value)

        return result

    @classmethod
    def from_dict(cls,data: dict[str, Any],) -> "Document":
        return cls(
            texto=str(data["texto"]),
            clase=data.get("clase"),
            fecha=data.get("fecha"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class DatasetDocument:

    dataset_id: str

    user_id: str

    filename: str

    created_at: str

    documents: list[Document]

    version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:

        return {
            "version": self.version,
            "dataset_id": self.dataset_id,
            "user_id": self.user_id,
            "filename": self.filename,
            "created_at": self.created_at,
            "documents": [
                document.to_dict()
                for document in self.documents
            ],
        }

    @classmethod
    def from_dict(cls,data: dict[str, Any],) -> "DatasetDocument":
        documents = [
            Document.from_dict(item)
            for item in data.get(
                "documents",
                [],
            )
        ]

        return cls(
            dataset_id=data["dataset_id"],
            user_id=data["user_id"],
            filename=data["filename"],
            created_at=data["created_at"],
            documents=documents,
            version=data.get(
                "version",
                "1.0",
            ),
        )


@dataclass
class DatasetInfo:

    dataset_id: str

    user_id: str

    filename: str

    total_documents: int

    created_at: str

@dataclass
class ChromaDocuments:

    ids: list[str]

    documents: list[str]

    metadatas: list[dict[str, Any]]

    embeddings: list[list[float]] | None = None