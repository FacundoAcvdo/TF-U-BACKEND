from ..cleaning_types import CleanerResult

class LowercaseCleaningService:

    def clean(self, text: str) -> CleanerResult:
        cleaned = text.lower()

        return CleanerResult(
            text=cleaned,
            count=1 if cleaned != text else 0,
        )