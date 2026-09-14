import re

from cleaning.cleaning_types import CleanerResult

URL_PATTERN = re.compile(
    r"https?://\S+|www\.\S+",
    re.IGNORECASE,
)


class UrlCleaningService:

    def clean(self, text: str) -> CleanerResult:
        matches = URL_PATTERN.findall(text)

        cleaned = URL_PATTERN.sub("", text)

        return CleanerResult(
            text=cleaned,
            count=len(matches),
        )