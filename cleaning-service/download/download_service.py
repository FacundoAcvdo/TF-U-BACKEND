import json
import shutil
import uuid

from datetime import (
    datetime,
    timezone,
)

from pathlib import Path

from download.download_types import (
    DownloadFile,
)


class DownloadService:

    CONTENT_TYPES = {
        ".csv": "text/csv",
        ".xlsx": (
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
    }

    def __init__(
        self,
        storage_path: str,
    ):

        self.storage_path = Path(
            storage_path
        )

        self.storage_path.mkdir(
            parents=True,
            exist_ok=True,
        )

    def create_download(
        self,
        source_path: Path,
        user_id: str,
        filename: str,
    ) -> DownloadFile:

        if not source_path.exists():
            raise FileNotFoundError(
                f"Archivo no encontrado: {source_path}"
            )

        extension = source_path.suffix.lower()

        if extension not in self.CONTENT_TYPES:
            raise ValueError(
                f"Formato no soportado: {extension}"
            )

        download_id = str(
            uuid.uuid4()
        )

        user_directory = (
            self.storage_path
            / user_id
        )

        user_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        destination = (
            user_directory
            / f"{download_id}{extension}"
        )

        shutil.copy2(
            source_path,
            destination,
        )

        created_at = datetime.now(
            timezone.utc
        )

        metadata = {
            "download_id": download_id,
            "user_id": user_id,
            "filename": filename,
            "content_type": self.CONTENT_TYPES[
                extension
            ],
            "created_at": created_at.isoformat(),
        }

        metadata_path = (
            user_directory
            / f"{download_id}.json"
        )

        metadata_path.write_text(
            json.dumps(
                metadata,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        return DownloadFile(
            download_id=download_id,
            user_id=user_id,
            file_path=destination,
            filename=filename,
            content_type=self.CONTENT_TYPES[
                extension
            ],
            created_at=created_at,
        )

    def get_download(
        self,
        download_id: str,
        user_id: str,
    ) -> DownloadFile:

        self._validate_download_id(
            download_id
        )

        user_directory = (
            self.storage_path
            / user_id
        )

        metadata_path = (
            user_directory
            / f"{download_id}.json"
        )

        if not metadata_path.exists():
            raise FileNotFoundError(
                "Descarga no encontrada."
            )

        metadata = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )

        filename = metadata["filename"]

        extension = Path(
            filename
        ).suffix.lower()

        file_path = (
            user_directory
            / f"{download_id}{extension}"
        )

        if not file_path.exists():
            raise FileNotFoundError(
                "Archivo de descarga no encontrado."
            )

        return DownloadFile(
            download_id=download_id,
            user_id=user_id,
            file_path=file_path,
            filename=filename,
            content_type=metadata[
                "content_type"
            ],
            created_at=datetime.fromisoformat(
                metadata["created_at"]
            ),
        )

    @staticmethod
    def _validate_download_id(
        download_id: str,
    ) -> None:

        try:
            uuid.UUID(download_id)

        except ValueError as exc:

            raise ValueError(
                "download_id inválido."
            ) from exc