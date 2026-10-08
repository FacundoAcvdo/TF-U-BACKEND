import pandas as pd
import numpy as np
from typing import Dict, List


def _compute_series_stats(series: pd.Series) -> Dict[str, float]:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    return {
        "count": int(s.count()),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1) if len(s) > 1 else 0.0),
        "min": float(s.min() if len(s) > 0 else 0.0),
        "max": float(s.max() if len(s) > 0 else 0.0),
        "median": float(s.median() if len(s) > 0 else 0.0),
        "q25": float(s.quantile(0.25) if len(s) > 0 else 0.0),
        "q75": float(s.quantile(0.75) if len(s) > 0 else 0.0),
        "sum": float(s.sum()),
    }


def get_corpus_emotion_stats(df: pd.DataFrame, emotions: List[str]) -> Dict[str, Dict[str, float]]:
    corpus_stats: Dict[str, Dict[str, float]] = {}
    for emo in emotions:
        if emo not in df.columns:
            corpus_stats[emo] = _compute_series_stats(pd.Series(dtype=float))
        else:
            corpus_stats[emo] = _compute_series_stats(df[emo])
    return corpus_stats


def get_class_emotion_stats(df: pd.DataFrame, emotions: List[str], class_column: str = "clase") -> List[Dict]:
    if class_column not in df.columns:
        raise ValueError(f"Columna de clase '{class_column}' no encontrada. Columnas: {list(df.columns)}")

    result: List[Dict] = []
    clases = df[class_column].dropna().unique()

    for clase in clases:
        subset = df[df[class_column] == clase]
        entry: Dict = {
            "clase": str(clase),
            "count": int(len(subset)),
        }
        for emo in emotions:
            if emo not in subset.columns:
                entry[emo] = _compute_series_stats(pd.Series(dtype=float))
            else:
                entry[emo] = _compute_series_stats(subset[emo])
        result.append(entry)

    result.sort(key=lambda x: x["clase"])
    return result
