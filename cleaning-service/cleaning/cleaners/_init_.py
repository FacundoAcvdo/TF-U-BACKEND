from dataclasses import dataclass


@dataclass
class CleanerResult:
    text: str
    count: int = 0