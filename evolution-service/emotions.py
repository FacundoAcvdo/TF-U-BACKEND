import numpy as np
import logging
from logging_utils import setup_service_logger
logger = setup_service_logger("evolution-service")


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
        if device == 0:
            try:
                model_kwargs["torch_dtype"] = torch.float16
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
        _pipeline = pipe
        pass
        return _pipeline
    except Exception as e:
        logger.warning(f"[evolution-service] fallo carga {MODEL_SINGLE}: {e} -> dummy")
        _pipeline = None
        return None


def predecir_emociones_textos(
    textos: list[str],
    categories: list[str] | None = None,
) -> np.ndarray:
    """Predice emociones de una lista de textos -> matriz (n, m) de puntajes 0-1."""
    if categories is None:
        categories = EMOTIONS
    n = len(textos)
    m = len(categories)
    if n == 0:
        return np.zeros((0, m), dtype=np.float32)

    truncados = [(t or "")[:2000] for t in textos]
    pipe = get_emotion_pipeline()

    if pipe is not None:
        try:
            results = pipe(truncados, truncation=True, max_length=512, batch_size=32)
            scores = np.zeros((n, m), dtype=np.float32)
            cat_index = {c: i for i, c in enumerate(categories)}
            for i, res in enumerate(results):
                if isinstance(res, dict):
                    res = [res]
                for item in res:
                    lab = str(item.get("label", "")).lower().strip()
                    if lab in cat_index:
                        scores[i, cat_index[lab]] = float(item.get("score", 0.0))
            return scores
        except Exception as e:
            logger.warning(f"[evolution-service] batch vectorizado fallo: {e} -> fallback por fila")
            scores = np.zeros((n, m), dtype=np.float32)
            cat_index = {c: i for i, c in enumerate(categories)}
            for i, t in enumerate(truncados):
                try:
                    res = pipe(t, truncation=True, max_length=512)
                    if isinstance(res, list) and len(res) > 0 and isinstance(res[0], list):
                        res = res[0]
                    if isinstance(res, dict):
                        res = [res]
                    for item in res:
                        lab = str(item.get("label", "")).lower().strip()
                        if lab in cat_index:
                            scores[i, cat_index[lab]] = float(item.get("score", 0.0))
                except Exception:
                    pass
            return scores

    logger.warning("[evolution-service] dummy scores (sin modelo)")
    dummy = np.random.rand(n, m).astype(np.float32)
    row_sums = dummy.sum(axis=1, keepdims=True)
    return dummy / np.where(row_sums == 0, 1, row_sums)


def promediar_emociones(
    scores: np.ndarray,
    categories: list[str] | None = None,
) -> dict[str, float]:
    """Promedia la matriz (n, m) por columna -> {emocion: puntaje medio 0-1}."""
    if categories is None:
        categories = EMOTIONS
    if scores.size == 0:
        return {c: 0.0 for c in categories}
    means = scores.mean(axis=0)
    return {c: float(means[i]) for i, c in enumerate(categories)}
