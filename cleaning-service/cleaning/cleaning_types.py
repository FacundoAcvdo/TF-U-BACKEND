from typing import Literal

from pydantic import BaseModel, Field


CleaningOperation = Literal[
    "lowercase",
    "remove_urls",
    "remove_mentions",
    "remove_html",
    "normalize_whitespace",
    "remove_duplicates",
    "handle_hashtags",
    "handle_emojis",
    "remove_punctuation",
    "remove_stopwords",
    "lemmatization",
    "anonymization",
]

EmojiStrategy = Literal[
    "keep",
    "remove",
    "convert",
]

HashtagStrategy = Literal[
    "keep",
    "remove_symbol",
    "segment",
]

Language = Literal[
    "en",
    "es",
    "bilingual",
]

class CleanerResult(BaseModel):
    text: str
    count: int = 0

class DownloadRequest(BaseModel):
    download_id: str

class CleaningRequest(BaseModel):
    dataset_name: str

    operations: list[CleaningOperation] = Field(
        default_factory=list
    )

    emoji_strategy: EmojiStrategy = "keep"

    hashtag_strategy: HashtagStrategy = "keep"

    language: Language = "bilingual"


class CleaningStats(BaseModel):
    original_rows: int = 0
    cleaned_rows: int = 0

    original_tokens: int = 0
    cleaned_tokens: int = 0

    original_unique_terms: int = 0
    cleaned_unique_terms: int = 0

    removed_duplicates: int = 0

    removed_urls: int = 0
    removed_mentions: int = 0
    removed_html: int = 0

    normalized_whitespace: int = 0

    emojis_processed: int = 0
    hashtags_processed: int = 0

    punctuation_removed: int = 0
    stopwords_removed: int = 0

    lemmas_processed: int = 0
    entities_anonymized: int = 0


class CleaningPreview(BaseModel):
    original: str
    cleaned: str


class CleaningResponse(BaseModel):
    source_dataset_id: str
    filename: str
    format: str

    download_id: str

    stats: CleaningStats

    previews: list[CleaningPreview]