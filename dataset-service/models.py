from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
from typing import Any
import math

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
            
            if isinstance(value, float) and math.isnan(value):
                continue

            if isinstance(value, (str, int, float, bool)):
                result[key] = value
            else:
                result[key] = str(value)

        if not result:
            result["_empty"] = True

        return result


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

    def to_json_bytes(self) -> bytes:
        return json.dumps(
            asdict(self),
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")


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