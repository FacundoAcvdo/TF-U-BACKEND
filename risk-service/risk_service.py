"""Cabeza de riesgo suicida (opción A): E5-large + LogReg calibrada.

Carga model_artifacts/risk_model.pkl una vez y expone scoring
vectorizado sobre los embeddings que ya están en Chroma.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

BASE_DIR = Path(__file__).parent
ARTIFACTS_DIR = BASE_DIR / "model_artifacts"
MODEL_PATH = ARTIFACTS_DIR / "risk_model.pkl"
THRESHOLDS_PATH = ARTIFACTS_DIR / "thresholds.json"


NIVELES = ["nulo", "bajo", "medio", "potencial", "alto"]

_bundle = None


def _load_bundle() -> dict:
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"No existe {MODEL_PATH}. Entrena primero con train_risk_head.py"
            )
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def get_thresholds() -> dict:
    bundle = _load_bundle()
    th = bundle.get("thresholds") or {}
    if THRESHOLDS_PATH.is_file():
        th = json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))
    return {
        "alto": float(th.get("alto", 0.85)),
        "potencial": float(th.get("potencial", 0.65)),
        "medio": float(th.get("medio", 0.40)),
        "bajo": float(th.get("bajo", 0.15)),
    }


def get_model_info() -> dict:
    bundle = _load_bundle()
    clf = bundle.get("clf")
    return {
        "embedding_model": bundle.get("embedding_model", "intfloat/multilingual-e5-large"),
        "embedding_dim": int(bundle.get("embedding_dim", 1024)),
        "trained_at": bundle.get("trained_at", ""),
        "classes": getattr(clf, "classes_", []).tolist()
        if hasattr(getattr(clf, "classes_", []), "tolist") else list(getattr(clf, "classes_", [])),
    }


def score_vectors(X: np.ndarray) -> list[float]:
    """P(suicidio) para cada fila. X: (n, dim) float32."""
    bundle = _load_bundle()
    proba = bundle["clf"].predict_proba(np.asarray(X, dtype=np.float32))[:, 1]
    return [float(p) for p in proba]


def score_to_level(p: float, th: dict | None = None) -> str:
    th = th or get_thresholds()
    if p >= th["alto"]:
        return "alto"
    if p >= th["potencial"]:
        return "potencial"
    if p >= th["medio"]:
        return "medio"
    if p >= th["bajo"]:
        return "bajo"
    return "nulo"


def warmup() -> dict:
    info = get_model_info()
    th = get_thresholds()

    dummy = np.zeros((1, info["embedding_dim"]), dtype=np.float32)
    _load_bundle()["clf"].predict_proba(dummy)
    return {**info, "thresholds": th}
