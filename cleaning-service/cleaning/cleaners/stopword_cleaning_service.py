import re

from cleaning.cleaning_types import CleanerResult

ENGLISH_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be",
    "by", "for", "from", "has", "he", "in",
    "is", "it", "its", "of", "on", "that",
    "the", "to", "was", "were", "will", "with",
    "i", "you", "your", "we", "they", "this",
    "these", "those", "or", "but", "if",
    "then", "there", "here", "what", "which",
    "who", "whom", "when", "where", "why",
    "how", "my", "our", "their", "me", "him",
    "her", "us", "them",
}


SPANISH_STOPWORDS = {
    "a", "al", "algo", "algunas", "algunos",
    "ante", "antes", "como", "con", "contra",
    "cual", "cuando", "de", "del", "desde",
    "donde", "dos", "el", "él", "ella",
    "ellas", "ellos", "en", "entre", "era",
    "erais", "eran", "eres", "es", "esa",
    "esas", "ese", "eso", "esos", "esta",
    "estas", "este", "esto", "estos",
    "fue", "ha", "han", "hasta", "hay",
    "la", "las", "le", "les", "lo", "los",
    "más", "me", "mi", "mis", "mucho",
    "muy", "nos", "nosotros", "o", "para",
    "pero", "por", "que", "qué", "se",
    "sea", "si", "sí", "sin", "sobre",
    "son", "su", "sus", "también", "te",
    "tiene", "tienen", "tu", "tus", "un",
    "una", "uno", "unos", "y", "ya",
    "yo",
}


class StopwordCleaningService:

    def __init__(self):
        self.stopwords = (
            ENGLISH_STOPWORDS
            | SPANISH_STOPWORDS
        )

    def clean(
        self,
        text: str,
    ) -> CleanerResult:

        words = re.findall(
            r"\S+",
            text,
        )

        cleaned_words: list[str] = []

        removed = 0

        for word in words:

            normalized = word.lower()

            if normalized in self.stopwords:
                removed += 1
                continue

            cleaned_words.append(word)

        return CleanerResult(
            text=" ".join(cleaned_words),
            count=removed,
        )
