"""Verificación con RoBERTa (opción B). Solo EN + score alto.

Modelo: vibhorag101/roberta-base-suicide-prediction-phr (EN-only, F1 ~0.96).
Carga lazy: si no hay GPU/internet o falla la descarga, verify() devuelve
None y el servicio sigue funcionando solo con la opción A.
"""
from __future__ import annotations

import os
import logging
from logging_utils import setup_service_logger
logger = setup_service_logger("risk-service")


VERIFY_MODEL_ID = os.environ.get(
    "VERIFY_MODEL_ID",
    "vibhorag101/roberta-base-suicide-prediction-phr",
)

_pipe = None
_load_error: str | None = None


_ES_MARKERS = (
    "á", "é", "í", "ó", "ú", "ñ", "¿", "¡",
    " que ", " los ", " las ", " una ", " para ", " porque ",
    " quiero ", " estoy ", " tengo ",
)


def is_english(text: str) -> bool:
    t = f" {(text or '').lower()} "
    return not any(m in t for m in _ES_MARKERS)


def _get_pipe():
    global _pipe, _load_error
    if _pipe is not None:
        return _pipe
    if _load_error is not None:
        return None
    try:
        import torch
        from transformers import pipeline

        device = 0 if torch.cuda.is_available() else -1
        _pipe = pipeline(
            "text-classification",
            model=VERIFY_MODEL_ID,
            top_k=None,
            device=device,
            truncation=True,
            max_length=512,
        )
        return _pipe
    except Exception as e:
        _load_error = str(e)
        logger.warning(f"[risk-service] verificación B no disponible: {e}")
        return None


def verify(texts: list[str], batch_size: int = 32) -> list[float | None]:
    """P(suicidio) según RoBERTa. None donde no se pudo verificar."""
    pipe = _get_pipe()
    if pipe is None or not texts:
        return [None] * len(texts)
    cleaned = [(t or "")[:2000] for t in texts]
    try:
        results = pipe(cleaned, truncation=True, max_length=512, batch_size=batch_size)
        out: list[float | None] = []
        for res in results:
            items = res if isinstance(res, list) else [res]
            best = 0.0
            for item in items:
                lab = str(item.get("label", "")).lower()
                s = float(item.get("score", 0.0))
                if "suicid" in lab or lab in ("label_1", "1"):
                    best = max(best, s)
                elif "non" in lab or "not" in lab or lab in ("label_0", "0"):
                    best = max(best, 1.0 - s)
                else:
                    best = max(best, s)
            out.append(best)
        return out
    except Exception as e:
        logger.exception(f"[risk-service] fallo verificación B: {e}")
        return [None] * len(texts)
