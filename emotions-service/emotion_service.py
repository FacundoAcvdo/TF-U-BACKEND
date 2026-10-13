import pandas as pd
import numpy as np
import re
import logging
from logging_utils import setup_service_logger
logger = setup_service_logger("emotions-service")


EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]


MODEL_SINGLE = "j-hartmann/emotion-english-distilroberta-base"

_pipeline = None

def _detect_device():
    try:
        import torch
        return 0 if torch.cuda.is_available() else -1
    except Exception:
        return -1

def get_emotion_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    device = _detect_device()
    pass
    try:
        from transformers import pipeline
        import torch

        model_kwargs = {}
        torch_dtype = None
        if device == 0:
            try:
                torch_dtype = torch.float16
                model_kwargs["torch_dtype"] = torch_dtype
            except Exception:
                pass
        pipe = pipeline(
            "text-classification",
            model=MODEL_SINGLE,
            top_k=None,
            device=device,
            truncation=True,
            max_length=512,
            model_kwargs=model_kwargs if model_kwargs else None,
        )


        pass
        _pipeline = pipe
        pass
        return _pipeline
    except Exception as e:
        logger.warning(f"[emotions-service] fallo carga {MODEL_SINGLE}: {e} -> dummy")
        _pipeline = None
        return None

def enrich_dataframe_with_emotions(
    df: pd.DataFrame,
    categories: list[str] = None,
    lexicon=None,
    text_column: str = "texto",
) -> pd.DataFrame:
    if categories is None:
        categories = EMOTIONS
    if text_column not in df.columns:
        raise ValueError(f"Columna '{text_column}' no encontrada. Columnas: {list(df.columns)}")

    pipe = get_emotion_pipeline()

    enriched = df.copy()
    for cat in categories:
        if cat not in enriched.columns:
            enriched[cat] = 0.0

    texts = enriched[text_column].astype(str).fillna("").tolist()

    texts = [t[:2000] for t in texts]

    if pipe is not None:
        try:

            results = pipe(texts, truncation=True, max_length=512, batch_size=32)

            n = len(texts)
            m = len(categories)
            scores_array = np.zeros((n, m), dtype=np.float32)
            cat_index = {c: i for i, c in enumerate(categories)}
            for i, res in enumerate(results):

                for item in res:
                    lab = str(item.get("label", "")).lower().strip()
                    if lab in cat_index:
                        scores_array[i, cat_index[lab]] = float(item.get("score", 0.0))

            enriched.loc[:, categories] = scores_array

            for cat in categories:
                enriched[cat] = pd.to_numeric(enriched[cat], errors="coerce").fillna(0.0).astype(float)
            return enriched
        except Exception as e:
            logger.warning(f"[emotions-service] batch vectorizado fallo: {e} -> fallback fila")

            for i, t in enumerate(texts):
                try:
                    res = pipe(t, truncation=True, max_length=512)
                    if isinstance(res, list) and len(res) > 0 and isinstance(res[0], list):
                        res = res[0]
                    scores = {c: 0.0 for c in categories}
                    for item in res:
                        lab = str(item.get("label","")).lower()
                        if lab in scores:
                            scores[lab] = float(item.get("score",0.0))
                    for cat in categories:
                        enriched.at[i, cat] = scores[cat]
                except Exception:
                    pass
            return enriched
    else:

        logger.warning("[emotions-service] dummy scores (sin modelo)")
        import random
        n = len(texts)
        m = len(categories)
        dummy = np.random.rand(n, m).astype(np.float32)

        row_sums = dummy.sum(axis=1, keepdims=True)
        dummy = dummy / np.where(row_sums==0, 1, row_sums)
        enriched.loc[:, categories] = dummy
        return enriched
