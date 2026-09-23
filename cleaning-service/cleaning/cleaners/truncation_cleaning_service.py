from cleaning.cleaning_types import CleanerResult

_SUPPORTED_LANGUAGES = {"en", "es"}


class TruncationCleaningService:

    _stemmer_cache: dict[str, object] = {}

    def _get_stemmer(self, language: str):
        """
        Devuelve el stemmer para el idioma dado, instanciándolo
        (y cacheándolo) solo la primera vez.
        """
        if language in self._stemmer_cache:
            return self._stemmer_cache[language]

        try:
            from nltk.stem import PorterStemmer, SnowballStemmer
        except ImportError as exc:
            raise RuntimeError(
                "nltk no está instalado."
            ) from exc

        if language == "en":
            stemmer = PorterStemmer()
        elif language == "es":
            stemmer = SnowballStemmer("spanish")
        else:
            raise ValueError(
                f"Idioma no soportado para truncado: {language}"
            )

        self._stemmer_cache[language] = stemmer
        return stemmer

    def _detect_language(self, text: str) -> str:
        """
        Detecta el idioma de una entrada individual, restringido
        a los idiomas soportados (en/es). Usa detect_langs para
        quedarse con el más probable entre los candidatos válidos,
        en vez de fallar si el resultado top no es en/es.
        """
        try:
            from langdetect import detect_langs, LangDetectException
        except ImportError as exc:
            raise RuntimeError(
                "langdetect no está instalado. Es requerido "
                "para truncado bilingüe o detección "
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
                f"No se detectó un idioma soportado para la "
                f"entrada. Candidatos detectados: "
                f"{[c.lang for c in candidates]}"
            )

        return supported_candidates[0].lang

    def clean(
        self,
        text: str,
        language: str | None = None,
    ) -> CleanerResult:

        if language is None:
            raise ValueError(
                "El truncado requiere especificar "
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
                f"Idioma no soportado para truncado: {language}"
            )

        try:
            from nltk.tokenize import word_tokenize
        except ImportError as exc:
            raise RuntimeError(
                "nltk no está instalado."
            ) from exc

        stemmer = self._get_stemmer(resolved_language)

        tokens = word_tokenize(text, language="spanish" if resolved_language == "es" else "english")

        stems = []
        processed = 0

        for token in tokens:
            stem = stemmer.stem(token)

            if stem != token:
                processed += 1

            stems.append(stem)

        return CleanerResult(
            text=" ".join(stems),
            count=processed,
        )