class DuplicateCleaningService:

    def clean(
        self,
        entries: list[dict],
    ) -> tuple[list[dict], int]:

        seen: set[str] = set()

        result: list[dict] = []

        removed = 0

        for entry in entries:
            document = entry["document"]

            text = str(
                document.get("texto") or ""
            )

            if text in seen:
                removed += 1
                continue

            seen.add(text)

            result.append(entry)

        return result, removed