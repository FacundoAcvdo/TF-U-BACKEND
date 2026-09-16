from cleaning.cleaning_types import CleanerResult

_SUPPORTED_LANGUAGES = {"en", "es"}

_MODEL_NAMES = {
    "en": "en_core_web_sm",
    "es": "es_core_news_sm",
}

class LemmatizationCleaningService:

    _nlp_cache: dict[str, "spacy.language.Language"] = {}

    def _get_nlp(self, language: str):
        """
        Devuelve el pipeline de spaCy para el idioma dado,
        cargándolo (y cacheándolo) solo la primera vez.
        """
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
            nlp = spacy.load(model_name)
        except OSError as exc:
            raise RuntimeError(
                f"No está instalado el modelo de spaCy "
                f"'{model_name}'."
            ) from exc

        self._nlp_cache[language] = nlp
        return nlp

    def _detect_language(self, text: str) -> str:

        try:
            from langdetect import detect_langs, LangDetectException
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
            c for c in candidates if c.lang in _SUPPORTED_LANGUAGES
        ]

        if not supported_candidates:
            raise ValueError(
                f"No se detectó un idioma soportado para la entrada. "
                f"Candidatos detectados: "
                f"{[c.lang for c in candidates]}"
            )

        # detect_langs ya viene ordenado de mayor a menor probabilidad
        return supported_candidates[0].lang

    def clean(
        self,
        text: str,
        language: str | None = None,
    ) -> CleanerResult:

        if language is None:
            raise ValueError(
                "La lematización requiere especificar "
                "el idioma ('en', 'es' o 'bilingual')."
            )

        if language == "bilingual":
            try:
                resolved_language = self._detect_language(text)
            except ValueError:
                return CleanerResult(text=text, count=0)
        elif language in _SUPPORTED_LANGUAGES:
            resolved_language = language
        else:
            raise ValueError(
                f"Idioma no soportado para lematización: {language}"
            )

        nlp = self._get_nlp(resolved_language)
        doc = nlp(text)

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