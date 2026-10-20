from cleaning.cleaning_types import CleanerResult


_SUPPORTED_LANGUAGES = {"en", "es"}

_MODEL_NAMES = {
    "en": "en_core_web_sm",
    "es": "es_core_news_sm",
}

_DEFAULT_BATCH_SIZE = 64


class LemmatizationCleaningService:

    _nlp_cache: dict[str, "spacy.language.Language"] = {}

    def _get_nlp(self, language: str):

        if language in self._nlp_cache:
            return self._nlp_cache[language]

        try:
            import spacy
        except ImportError as exc:
            raise RuntimeError(
                "spaCy no está instalado."
            ) from exc

        model_name = _MODEL_NAMES[language]

        try:
            nlp = spacy.load(
                model_name,
                disable=["parser", "ner"],
            )
        except OSError as exc:
            raise RuntimeError(
                f"No está instalado el modelo de spaCy "
                f"'{model_name}'."
            ) from exc

        self._nlp_cache[language] = nlp

        return nlp

    def _detect_language(self, text: str) -> str:

        try:
            from langdetect import (
                detect_langs,
                LangDetectException,
            )
        except ImportError as exc:
            raise RuntimeError(
                "langdetect no está instalado. Es requerido "
                "para lematización bilingüe o detección "
                "automática de idioma."
            ) from exc

        if not text.strip():
            raise ValueError(
                "No se pudo detectar el idioma: texto vacío."
            )

        try:
            candidates = detect_langs(text)
        except LangDetectException as exc:
            raise ValueError(
                "No se pudo detectar el idioma de la entrada "
                "(texto demasiado corto o ambiguo)."
            ) from exc

        supported_candidates = [
            candidate
            for candidate in candidates
            if candidate.lang in _SUPPORTED_LANGUAGES
        ]

        if not supported_candidates:
            raise ValueError(
                "No se detectó un idioma soportado para la entrada. "
                f"Candidatos detectados: "
                f"{[candidate.lang for candidate in candidates]}"
            )

        return supported_candidates[0].lang

    @staticmethod
    def _process_doc(doc) -> CleanerResult:
        """
        Convierte un Doc de spaCy en CleanerResult.
        """
        lemmas = []
        processed = 0

        for token in doc:
            lemma = token.lemma_

            if lemma != token.text:
                processed += 1

            lemmas.append(lemma)

        return CleanerResult(
            text=" ".join(lemmas),
            count=processed,
        )

    def _clean_batch_for_language(
        self,
        texts: list[str],
        language: str,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> list[CleanerResult]:

        if not texts:
            return []

        nlp = self._get_nlp(language)

        results = []

        for doc in nlp.pipe(
            texts,
            batch_size=batch_size,
        ):
            results.append(
                self._process_doc(doc)
            )

        return results

    def clean_batch(
        self,
        texts: list[str],
        language: str | None = None,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> list[CleanerResult]:

        if language is None:
            raise ValueError(
                "La lematización requiere especificar "
                "el idioma ('en', 'es' o 'bilingual')."
            )

        if language not in _SUPPORTED_LANGUAGES and language != "bilingual":
            raise ValueError(
                f"Idioma no soportado para lematización: {language}"
            )

        if not texts:
            return []

        if language in _SUPPORTED_LANGUAGES:
            return self._clean_batch_for_language(
                texts=texts,
                language=language,
                batch_size=batch_size,
            )

        results: list[CleanerResult | None] = [
            None
        ] * len(texts)

        language_groups: dict[str, list[int]] = {
            "en": [],
            "es": [],
        }

        for index, text in enumerate(texts):

            if not text.strip():
                results[index] = CleanerResult(
                    text=text,
                    count=0,
                )
                continue

            try:
                resolved_language = self._detect_language(
                    text
                )
            except ValueError:
                results[index] = CleanerResult(
                    text=text,
                    count=0,
                )
                continue

            language_groups[
                resolved_language
            ].append(index)

        english_indexes = language_groups["en"]

        if english_indexes:

            english_texts = [
                texts[index]
                for index in english_indexes
            ]

            english_results = (
                self._clean_batch_for_language(
                    texts=english_texts,
                    language="en",
                    batch_size=batch_size,
                )
            )

            for index, result in zip(
                english_indexes,
                english_results,
            ):
                results[index] = result

        spanish_indexes = language_groups["es"]

        if spanish_indexes:

            spanish_texts = [
                texts[index]
                for index in spanish_indexes
            ]

            spanish_results = (
                self._clean_batch_for_language(
                    texts=spanish_texts,
                    language="es",
                    batch_size=batch_size,
                )
            )

            for index, result in zip(
                spanish_indexes,
                spanish_results,
            ):
                results[index] = result

        return [
            result
            if result is not None
            else CleanerResult(
                text=text,
                count=0,
            )
            for text, result in zip(
                texts,
                results,
            )
        ]

    def clean(
        self,
        text: str,
        language: str | None = None,
    ) -> CleanerResult:
        return self.clean_batch(
            texts=[text],
            language=language,
        )[0]
