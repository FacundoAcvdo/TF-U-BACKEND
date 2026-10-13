from pathlib import Path

import pandas as pd

from chroma_reader import fetch_paginated

_INTERNAL_METADATA_FIELDS = {
    "formato_original",
    "formato_detectado",
}


_CHROMA_INTERNAL_KEYS = {
    "dataset_id",
    "user_id",
    "source_filename",
    "_empty",
    "_v",
}

class ChromaDatasetService:

    def __init__(self) -> None:
        pass

    async def load_dataset(
        self,
        user_id: str,
        dataset_id: str,
    ) -> dict:
        where = {
            "$and": [
                {"user_id": user_id},
                {"dataset_id": dataset_id},
            ]
        }
        data = await fetch_paginated(where, ["documents", "metadatas"])

        ids = data.get("ids") or []
        if not ids:
            raise FileNotFoundError(
                "Dataset no encontrado."
            )

        documents_raw = data.get("documents", [])
        metadatas_raw = data.get("metadatas", [])

        filename = None
        documents: list[dict] = []
        for i, texto in enumerate(documents_raw):
            meta = dict(metadatas_raw[i]) if i < len(metadatas_raw) else {}

            if filename is None and meta.get("source_filename"):
                filename = str(meta.get("source_filename"))

            clase = meta.pop("clase", None)
            fecha = meta.pop("fecha", None)
            for k in _CHROMA_INTERNAL_KEYS:
                meta.pop(k, None)

            documents.append(
                {
                    "texto": texto,
                    "clase": clase,
                    "fecha": fecha,
                    "metadata": meta,
                }
            )

        return {
            "filename": filename or f"{dataset_id}.csv",
            "documents": documents,
        }

    def export_cleaned_dataset(
        self,
        documents: list[dict],
        output_path: Path,
    ) -> None:

        rows = []

        for document in documents:

            row = {
                "texto": document.get("texto"),
                "clase": document.get("clase"),
            }

            fecha = document.get("fecha")

            if fecha is not None:
                row["fecha"] = fecha

            metadata = document.get(
                "metadata",
                {},
            )

            if isinstance(metadata, dict):
                row.update(
                    {
                        key: value
                        for key, value in metadata.items()
                        if key not in _INTERNAL_METADATA_FIELDS
                    }
                )

            rows.append(row)

        dataframe = pd.DataFrame(rows)

        extension = output_path.suffix.lower()

        if extension == ".csv":

            dataframe.to_csv(
                output_path,
                index=False,
                encoding="utf-8-sig",
            )

        elif extension == ".xlsx":

            dataframe.to_excel(
                output_path,
                index=False,
                engine="openpyxl",
            )

        else:
            raise ValueError(
                f"Formato no soportado: {extension}"
            )

    @staticmethod
    def get_original_filename(
        dataset: dict,
    ) -> str:

        filename = dataset.get("filename")

        if filename:
            return filename

        documents = dataset.get(
            "documents",
            [],
        )

        if documents:

            metadata = documents[0].get(
                "metadata",
                {},
            )

            original_format = metadata.get(
                "formato_original"
            )

            if original_format:
                return (
                    f"cleaned.{original_format}"
                )

        raise ValueError(
            "No se pudo determinar el formato original."
        )
