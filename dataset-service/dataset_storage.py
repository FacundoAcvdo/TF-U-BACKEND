import json
from abc import ABC, abstractmethod
from pathlib import Path

from models import DatasetDocument


class DatasetStorage(ABC):

    @abstractmethod
    async def save(
        self,
        dataset: DatasetDocument,
    ) -> None:

        raise NotImplementedError

class LocalJsonDatasetStorage(
    DatasetStorage
):

    def __init__(
        self,
        base_path: Path,
    ) -> None:

        self.base_path = Path(base_path)

        self.base_path.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _dataset_path(
        self,
        user_id: str,
        dataset_id: str,
    ) -> Path:

        return (
            self.base_path
            / "users"
            / user_id
            / f"{dataset_id}.json"
        )

    async def save(
        self,
        dataset: DatasetDocument,
    ) -> None:

        path = self._dataset_path(
            user_id=dataset.user_id,
            dataset_id=dataset.dataset_id,
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path = path.with_suffix(
            ".tmp"
        )

        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                dataset.to_dict(),
                file,
                ensure_ascii=False,
                indent=2,
            )
        temporary_path.replace(
            path
        )

    async def delete_dataset(
        self,
        user_id: str,
        dataset_id: str,
    ) -> None:

        path = self._dataset_path(
            user_id=user_id,
            dataset_id=dataset_id,
        )

        path.unlink(
            missing_ok=True
        )

    async def delete_datasets_by_user(
        self,
        user_id: str,
    ) -> None:

        user_path = (
            self.base_path
            / "users"
            / user_id
        )

        for dataset_file in user_path.glob("*.json"):
            dataset_file.unlink(
                missing_ok=True
            )

        user_path.rmdir(
            missing_ok=True
        )