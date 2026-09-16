import asyncio
import json
from pathlib import Path
from typing import Any, Optional


class StorageService:

    def __init__(self, base_dir: str = "./data/topics") -> None:
        self.base_path = Path(base_dir)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_file_path(self, user_id: str, dataset_id: str, max_topics: int) -> Path:
        user_dir = self.base_path / str(user_id)
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir / f"{dataset_id}_{max_topics}_topics.json"

    async def get_topics(
        self, user_id: str, dataset_id: str, max_topics: int
    ) -> Optional[dict[str, Any]]:
        return await asyncio.to_thread(
            self._get_topics_sync, user_id, dataset_id, max_topics
        )

    def _get_topics_sync(
        self, user_id: str, dataset_id: str, max_topics: int
    ) -> Optional[dict[str, Any]]:
        file_path = self._get_file_path(user_id, dataset_id, max_topics)
        if not file_path.is_file():
            return None
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    async def save_topics(
        self, user_id: str, dataset_id: str, max_topics: int, data: dict[str, Any]
    ) -> None:
        await asyncio.to_thread(
            self._save_topics_sync, user_id, dataset_id, max_topics, data
        )

    def _save_topics_sync(
        self, user_id: str, dataset_id: str, max_topics: int, data: dict[str, Any]
    ) -> None:
        file_path = self._get_file_path(user_id, dataset_id, max_topics)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)