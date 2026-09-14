from cleaning.cleaning_types import CleanerResult

class LemmatizationCleaningService:

    def clean(
        self,
        text: str,
        language: str | None = None,
    ) -> CleanerResult:

        if language is None:
            raise ValueError(
                "La lematización requiere especificar "
                "el idioma ('en' o 'es'). "
                "Para datasets bilingües se requiere "
                "detección de idioma por documento."
            )

        if language == "bilingual":
            raise ValueError(
                "La lematización bilingüe todavía no está "
                "implementada. Se requiere detección de "
                "idioma por documento."
            )

        if language not in {"en", "es"}:
            raise ValueError(
                f"Idioma no soportado para lematización: {language}"
            )

        try:
            import spacy
        except ImportError as exc:
            raise RuntimeError(
                "spaCy no está instalado."
            ) from exc

        model_name = (
            "en_core_web_sm"
            if language == "en"
            else "es_core_news_sm"
        )

        try:
            nlp = spacy.load(model_name)
        except OSError as exc:
            raise RuntimeError(
                f"No está instalado el modelo de spaCy "
                f"'{model_name}'."
            ) from exc

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