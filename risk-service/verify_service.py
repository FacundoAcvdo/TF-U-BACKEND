"""Verificación con RoBERTa (opción B). Solo EN + score alto.

Modelo: vibhorag101/roberta-base-suicide-prediction-phr (EN-only, F1 ~0.96).
Carga lazy: si no hay GPU/internet o falla la descarga, verify() devuelve
None y el servicio sigue funcionando solo con la opción A.
"""
from __future__ import annotations

import os
import threading
import logging
from logging_utils import setup_service_logger

import model_cache
from model_cache import ModelBusyError, ModelSlot
logger = setup_service_logger("risk-service")


VERIFY_MODEL_ID = os.environ.get(
    "VERIFY_MODEL_ID",
    "vibhorag101/roberta-base-suicide-prediction-phr",
)

VERIFY_VRAM_MB = int(os.environ.get("VERIFY_VRAM_MB", "1500"))

_ES_MARKERS = (
    "á", "é", "í", "ó", "ú", "ñ", "¿", "¡",
    " que ", " los ", " las ", " una ", " para ", " porque ",
    " quiero ", " estoy ", " tengo ",
)


def is_english(text: str) -> bool:
    t = f" {(text or '').lower()} "
    return not any(m in t for m in _ES_MARKERS)


_slot = None
_slot_lock = threading.Lock()


def _load_pipe():
    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "text-classification",
        model=VERIFY_MODEL_ID,
        top_k=None,
        device=device,
        truncation=True,
        max_length=512,
    )


def _drop_pipe(pipe):
    try:
        del pipe
    except Exception:
        pass


def _get_slot():
    global _slot
    slot = _slot
    if slot is None:
        with _slot_lock:
            slot = _slot
            if slot is None:
                slot = ModelSlot(name="risk-verifier", vram_mb=VERIFY_VRAM_MB,
                                 load_fn=_load_pipe, unload_fn=_drop_pipe)
                _slot = slot
    return slot


def _get_pipe():
    slot = _get_slot()
    slot.maybe_evict()
    try:
        return slot.get()
    except ModelBusyError as e:
        logger.warning(f"[risk-service] verificacion B omitida sin VRAM: {e}")
        return None
    except Exception as e:
        logger.warning(f"[risk-service] verificación B no disponible: {e}")
        return None


def verify(texts: list[str], batch_size: int = 32) -> list[float | None]:
    """P(suicidio) según RoBERTa. None donde no se pudo verificar."""
    if not texts:
        return []
    slot = _get_slot()
    pipe = _get_pipe()
    if pipe is None:
        return [None] * len(texts)
    try:
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
    finally:
        try:
            slot.release()
        except Exception:
            pass
