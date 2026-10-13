import re

from cleaning.cleaning_types import CleanerResult

MENTION_PATTERN = re.compile(
    r"(?<!\w)@\w+",
)


class MentionCleaningService:

    def clean(self, text: str) -> CleanerResult:
        matches = MENTION_PATTERN.findall(text)

        cleaned = MENTION_PATTERN.sub("", text)

        return CleanerResult(
            text=cleaned,
            count=len(matches),
        )
