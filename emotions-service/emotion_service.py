import pandas as pd
from empath import Empath

# Categorías fijas solicitadas por el requerimiento
EMOTIONS = [
    "anger",
    "violence",
    "sadness",
    "negative_emotion",
    "fear",
    "fun",
    "joy",
    "positive_emotion",
    "optimism",
    "trust",
]


def get_empath_client() -> Empath:
    return Empath()


def analyze_text_emotions(text: str, lexicon: Empath, categories: list[str] = None) -> dict:
    if categories is None:
        categories = EMOTIONS

    if text is None or (isinstance(text, float) and pd.isna(text)):
        text = ""
    else:
        text = str(text)

    scores = lexicon.analyze(text, categories=categories, normalize=True)

    return {cat: float(scores.get(cat, 0.0)) for cat in categories}


def enrich_dataframe_with_emotions(
    df: pd.DataFrame,
    categories: list[str] = None,
    lexicon: Empath = None,
    text_column: str = "texto",
) -> pd.DataFrame:
    if categories is None:
        categories = EMOTIONS

    if lexicon is None:
        lexicon = get_empath_client()

    if text_column not in df.columns:
        raise ValueError(f"Columna '{text_column}' no encontrada en el DataFrame. Columnas: {list(df.columns)}")

    enriched = df.copy()

    for cat in categories:
        if cat not in enriched.columns:
            enriched[cat] = 0.0

    for idx, row in enriched.iterrows():
        scores = analyze_text_emotions(row[text_column], lexicon, categories)
        for cat in categories:
            enriched.at[idx, cat] = scores[cat]

    for cat in categories:
        enriched[cat] = enriched[cat].astype(float)

    return enriched
