import json
from pathlib import Path

import pandas as pd

_INTERNAL_METADATA_FIELDS = {
    "formato_original",
    "formato_detectado",
}

class DatasetStorageService:

    def __init__(
        self,
        dataset_storage_path: str,
    ):
        self.dataset_storage_path = Path(
            dataset_storage_path
        )

    def get_dataset_path(
        self,
        user_id: str,
        dataset_id: str,
    ) -> Path:

        return (
            self.dataset_storage_path
            / "users"
            / user_id
            / f"{dataset_id}.json"
        )

    def load_dataset(
        self,
        user_id: str,
        dataset_id: str,
    ) -> dict:

        path = self.get_dataset_path(
            user_id,
            dataset_id,
        )

        if not path.exists():
            raise FileNotFoundError(
                "Dataset no encontrado."
            )

        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

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