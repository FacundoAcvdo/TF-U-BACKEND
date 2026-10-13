import re

from cleaning.cleaning_types import CleanerResult

WHITESPACE_PATTERN = re.compile(
    r"\s+",
)


class WhitespaceCleaningService:

    def clean(self, text: str) -> CleanerResult:
        cleaned = WHITESPACE_PATTERN.sub(" ", text).strip()

        return CleanerResult(
            text=cleaned,
            count=1 if cleaned != text else 0,
        )
