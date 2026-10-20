import copy
import tempfile

from pathlib import Path
from urllib import request

from .cleaning_types import (
    CleaningRequest,
    CleaningResponse,
    CleaningStats,
    CleaningPreview,
)

from .cleaners.lowercase_cleaning_service import (
    LowercaseCleaningService,
)

from .cleaners.url_cleaning_service import (
    UrlCleaningService,
)

from .cleaners.mention_cleaning_service import (
    MentionCleaningService,
)

from .cleaners.truncation_cleaning_service import (
    TruncationCleaningService,
)

from .cleaners.whitespace_cleaning_service import (
    WhitespaceCleaningService,
)

from .cleaners.duplicate_cleaning_service import (
    DuplicateCleaningService,
)

from .cleaners.hashtag_cleaning_service import (
    HashtagCleaningService,
)

from .cleaners.emoji_cleaning_service import (
    EmojiCleaningService,
)

from .cleaners.punctuation_cleaning_service import (
    PunctuationCleaningService,
)

from .cleaners.stopword_cleaning_service import (
    StopwordCleaningService,
)

from .cleaners.lemmatization_service import (
    LemmatizationCleaningService,
)

from .cleaners.anonymization_service import (
    AnonymizationCleaningService,
)


class CleaningService:

    TEXT_FIELD = "texto"

    MAX_PREVIEWS = 10

    def __init__(
        self,
        dataset_service,
        supabase_service,
        download_service,
    ):

        self.dataset_service = (
            dataset_service
        )

        self.download_service = (
            download_service
        )

        self.supabase_service = (
            supabase_service
        )

        self.cleaners = {
            "lowercase": (
                LowercaseCleaningService()
            ),
            "remove_urls": (
                UrlCleaningService()
            ),
            "remove_mentions": (
                MentionCleaningService()
            ),
            "truncation": (
                TruncationCleaningService()
            ),
            "normalize_whitespace": (
                WhitespaceCleaningService()
            ),
            "handle_hashtags": (
                HashtagCleaningService()
            ),
            "handle_emojis": (
                EmojiCleaningService()
            ),
            "remove_punctuation": (
                PunctuationCleaningService()
            ),
            "remove_stopwords": (
                StopwordCleaningService()
            ),
            "lemmatization": (
                LemmatizationCleaningService()
            ),
            "anonymization": (
                AnonymizationCleaningService()
            ),
        }

        self.duplicate_cleaner = (
            DuplicateCleaningService()
        )

    async def clean(
        self,
        request: CleaningRequest,
        user_id: str,
        user_token: str,
    ) -> CleaningResponse:

        dataset_id = await self.supabase_service.get_dataset_id_by_name(
            dataset_name=request.dataset_name,
            user_id=user_id,
            user_token=user_token,
        )

        dataset = await (
            self.dataset_service.load_dataset(
                user_id=user_id,
                dataset_id=dataset_id,
            )
        )

        original_documents = dataset.get(
            "documents",
            [],
        )

        entries = []

        for document in original_documents:

            copied_document = copy.deepcopy(
                document
            )

            original_text = str(
                copied_document.get(
                    self.TEXT_FIELD
                )
                or ""
            )

            entries.append(
                {
                    "document": copied_document,
                    "original_text": original_text,
                }
            )

        stats = CleaningStats()

        stats.original_rows = len(
            entries
        )

        stats.original_tokens = (
            self._count_tokens(
                [
                    entry["original_text"]
                    for entry in entries
                ]
            )
        )

        stats.original_unique_terms = (
            self._count_unique_terms(
                [
                    entry["original_text"]
                    for entry in entries
                ]
            )
        )

        for operation in request.operations:

            if operation == "remove_duplicates":

                entries, removed = (
                    self.duplicate_cleaner.clean(
                        entries
                    )
                )

                stats.removed_duplicates += (
                    removed
                )

                continue

            cleaner = self.cleaners.get(
                operation
            )

            if cleaner is None:
                raise ValueError(
                    f"Operación no soportada: "
                    f"{operation}"
                )

            if operation == "lemmatization":

                texts = [
                    str(
                        entry["document"].get(
                            self.TEXT_FIELD
                        )
                        or ""
                    )
                    for entry in entries
                ]

                results = cleaner.clean_batch(
                    texts=texts,
                    language=request.language,
                )

                total_count = sum(
                    result.count
                    for result in results
                )

                for entry, result in zip(
                    entries,
                    results,
                ):
                    entry["document"][
                        self.TEXT_FIELD
                    ] = result.text

                self._update_stats(
                    stats=stats,
                    operation=operation,
                    count=total_count,
                )

                continue

            for entry in entries:

                document = entry[
                    "document"
                ]

                text = str(
                    document.get(
                        self.TEXT_FIELD
                    )
                    or ""
                )

                if operation == "handle_emojis":

                    result = cleaner.clean(
                        text,
                        request.emoji_strategy,
                    )

                elif operation == "handle_hashtags":

                    result = cleaner.clean(
                        text,
                        request.hashtag_strategy,
                    )

                elif operation == "truncation":

                    result = cleaner.clean(
                        text,
                        request.language,
                    )

                else:

                    result = cleaner.clean(
                        text
                    )

                document[
                    self.TEXT_FIELD
                ] = result.text

                self._update_stats(
                    stats=stats,
                    operation=operation,
                    count=result.count,
                )

        final_texts = [
            str(
                entry["document"].get(
                    self.TEXT_FIELD
                )
                or ""
            )
            for entry in entries
        ]

        stats.cleaned_rows = len(
            entries
        )

        stats.cleaned_tokens = (
            self._count_tokens(
                final_texts
            )
        )

        stats.cleaned_unique_terms = (
            self._count_unique_terms(
                final_texts
            )
        )

        previews = []

        for entry in entries:

            original_text = entry[
                "original_text"
            ]

            cleaned_text = str(
                entry["document"].get(
                    self.TEXT_FIELD
                )
                or ""
            )

            if original_text == cleaned_text:
                continue

            previews.append(
                CleaningPreview(
                    original=original_text,
                    cleaned=cleaned_text,
                )
            )

            if len(previews) >= self.MAX_PREVIEWS:
                break

        original_filename = (
            self.dataset_service
            .get_original_filename(
                dataset
            )
        )

        extension = Path(
            original_filename
        ).suffix.lower()

        if extension not in {
            ".csv",
            ".xlsx",
        }:

            raise ValueError(
                "El dataset debe tener formato "
                "CSV o XLSX."
            )

        cleaned_filename = (
            f"cleaned_{Path(original_filename).name}"
        )

        with tempfile.TemporaryDirectory() as temp_dir:

            output_path = (
                Path(temp_dir)
                / cleaned_filename
            )

            documents = [
                entry["document"]
                for entry in entries
            ]

            self.dataset_service.export_cleaned_dataset(
                documents=documents,
                output_path=output_path,
            )

            download = (
                self.download_service.create_download(
                    source_path=output_path,
                    user_id=user_id,
                    filename=cleaned_filename,
                )
            )

        return CleaningResponse(
            source_dataset_id=dataset_id,
            filename=cleaned_filename,
            format=extension.lstrip("."),
            download_id=download.download_id,
            stats=stats,
            previews=previews,
        )

    @staticmethod
    def _count_tokens(
        texts: list[str],
    ) -> int:

        return sum(
            len(text.split())
            for text in texts
        )

    @staticmethod
    def _count_unique_terms(
        texts: list[str],
    ) -> int:

        terms = set()

        for text in texts:

            terms.update(
                word.lower()
                for word in text.split()
                if word
            )

        return len(terms)

    @staticmethod
    def _update_stats(
        stats: CleaningStats,
        operation: str,
        count: int,
    ) -> None:

        if operation == "lowercase":
            return

        if operation == "remove_urls":
            stats.removed_urls += count

        elif operation == "remove_mentions":
            stats.removed_mentions += count

        elif operation == "normalize_whitespace":
            stats.normalized_whitespace += count

        elif operation == "handle_emojis":
            stats.emojis_processed += count

        elif operation == "handle_hashtags":
            stats.hashtags_processed += count

        elif operation == "remove_punctuation":
            stats.punctuation_removed += count

        elif operation == "remove_stopwords":
            stats.stopwords_removed += count

        elif operation == "lemmatization":
            stats.lemmas_processed += count

        elif operation == "truncation":
            stats.truncated_words += count

        elif operation == "anonymization":
            stats.entities_anonymized += count
