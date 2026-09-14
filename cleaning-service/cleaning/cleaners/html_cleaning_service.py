import re

from cleaning.cleaning_types import CleanerResult

HTML_PATTERN = re.compile(
    r"<[^>]+>",
)


class HtmlCleaningService:

    def clean(self, text: str) -> CleanerResult:
        matches = HTML_PATTERN.findall(text)

        cleaned = HTML_PATTERN.sub("", text)

        return CleanerResult(
            text=cleaned,
            count=len(matches),
        )