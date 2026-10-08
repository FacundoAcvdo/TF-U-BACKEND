import string

from cleaning.cleaning_types import CleanerResult

class PunctuationCleaningService:

    def clean(self, text: str) -> CleanerResult:

        punctuation = set(string.punctuation)

        removed = sum(
            1
            for char in text
            if char in punctuation
        )

        cleaned = "".join(
            char
            for char in text
            if char not in punctuation
        )

        return CleanerResult(
            text=cleaned,
            count=removed,
        )