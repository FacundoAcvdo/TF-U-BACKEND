from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class DownloadFile:

    download_id: str

    user_id: str

    file_path: Path

    filename: str

    content_type: str

    created_at: datetime
