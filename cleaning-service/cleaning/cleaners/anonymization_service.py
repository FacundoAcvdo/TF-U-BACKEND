import re

from cleaning.cleaning_types import CleanerResult


EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

PHONE_PATTERN = re.compile(
    r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)"
)

IP_PATTERN = re.compile(
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
)


class AnonymizationCleaningService:

    def clean(self, text: str) -> CleanerResult:

        total = 0

        text, count = EMAIL_PATTERN.subn(
            "[EMAIL]",
            text,
        )
        total += count

        text, count = PHONE_PATTERN.subn(
            "[PHONE]",
            text,
        )
        total += count

        text, count = IP_PATTERN.subn(
            "[IP]",
            text,
        )
        total += count

        return CleanerResult(
            text=text,
            count=total,
        )