import re

from cleaning.cleaning_types import CleanerResult

HASHTAG_PATTERN = re.compile(
    r"#(\w+)",
)


class HashtagCleaningService:

    def clean(
        self,
        text: str,
        strategy: str,
    ) -> CleanerResult:

        matches = HASHTAG_PATTERN.findall(text)

        if strategy == "keep":
            return CleanerResult(
                text=text,
                count=0,
            )

        if strategy == "remove_symbol":
            cleaned = HASHTAG_PATTERN.sub(
                r"\1",
                text,
            )

            return CleanerResult(
                text=cleaned,
                count=len(matches),
            )

        if strategy == "segment":

            def segment(match: re.Match) -> str:
                hashtag = match.group(1)

                hashtag = hashtag.replace("_", " ")

                hashtag = re.sub(
                    r"(?<=[a-z])(?=[A-Z])",
                    " ",
                    hashtag,
                )

                return hashtag

            cleaned = HASHTAG_PATTERN.sub(
                segment,
                text,
            )

            return CleanerResult(
                text=cleaned,
                count=len(matches),
            )

        raise ValueError(
            f"Estrategia de hashtag no soportada: {strategy}"
        )
