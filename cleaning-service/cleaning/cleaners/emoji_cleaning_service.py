import re

import emoji

from cleaning.cleaning_types import CleanerResult

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "]"
)


class EmojiCleaningService:

    def clean(
        self,
        text: str,
        strategy: str,
    ) -> CleanerResult:

        matches = EMOJI_PATTERN.findall(text)

        if strategy == "keep":
            return CleanerResult(
                text=text,
                count=0,
            )

        if strategy == "remove":
            cleaned = EMOJI_PATTERN.sub(
                "",
                text,
            )

            return CleanerResult(
                text=cleaned,
                count=len(matches),
            )

        if strategy == "convert":
            cleaned = emoji.demojize(
                text,
                language="en",
            )

            return CleanerResult(
                text=cleaned,
                count=len(matches),
            )

        raise ValueError(
            f"Estrategia de emoji no soportada: {strategy}"
        )
