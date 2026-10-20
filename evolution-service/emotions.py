"""Scoring de emociones bilingue ES/EN.

COPIAS IDENTICAS en emotions-service/emotion_service.py y
evolution-service/emotions.py (levantan por separado): mantener
sincronizadas a mano.

Diseno:
  - Router por idioma: ES -> pysentimiento/robertuito-base-uncased-emotion,
    EN (y resto) -> j-hartmann/emotion-english-distilroberta-base.
  - Salida unificada: 6 emociones de Ekman + neutral interno. Las etiquetas
    extra de cada modelo se ignoran (argmax restringido al set compartido).
  - Preproceso de chat (prefijos, URLs, menciones, alargamientos, emojis).
  - Ventana deslizante para textos largos en vez de truncar la cabeza.
  - Sin "dummy": si un modelo no carga, se lanza RuntimeError (el job falla
    con error HTTP y el frontend avisa; nunca se devuelven scores random).
"""
import logging
import re
import threading

import numpy as np

import model_cache
from model_cache import ModelBusyError, ModelSlot

logger = logging.getLogger("emotion-scoring")


EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
NEUTRAL = "neutral"
TARGETS = EMOTIONS + [NEUTRAL]

MODEL_ES = "pysentimiento/robertuito-base-uncased-emotion"
MODEL_EN = "j-hartmann/emotion-english-distilroberta-base"

LOW_CONF_THRESHOLD = 0.40
BATCH_SIZE = 32
MAX_WORDS_WINDOW = 380
WINDOW_OVERLAP = 80
MAX_WINDOWS = 4
SHORT_TEXT_CHARS = 15

_PREFIX_RE = re.compile(r"^\[[^\]]{0,40}\]\s*[^:\n]{1,60}:\s*")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_REPEAT_RE = re.compile(r"(.)\1{2,}")

_slots = {}
_slots_lock = threading.Lock()
_SLOT_VRAM_MB = {"es": 1200, "en": 800}


def _make_slot(lang, model_id):
    def _load():
        return _load_pipe(model_id)

    def _unload(pipe):
        try:
            del pipe
        except Exception:
            pass

    return ModelSlot(name=f"emociones-{lang}", vram_mb=_SLOT_VRAM_MB[lang],
                     load_fn=_load, unload_fn=_unload)


def _slot(lang):
    slot = _slots.get(lang)
    if slot is None:
        with _slots_lock:
            slot = _slots.get(lang)
            if slot is None:
                slot = _make_slot(lang, MODEL_ES if lang == "es" else MODEL_EN)
                _slots[lang] = slot
    return slot


def _detect_device():
    try:
        import torch
        return 0 if torch.cuda.is_available() else -1
    except Exception:
        return -1


def _load_pipe(model_id):
    from transformers import pipeline
    device = _detect_device()
    model_kwargs = None
    if device == 0:
        try:
            import torch
            model_kwargs = {"torch_dtype": torch.float16}
        except Exception:
            model_kwargs = None
    return pipeline(
        "text-classification",
        model=model_id,
        top_k=None,
        device=device,
        truncation=True,
        max_length=512,
        model_kwargs=model_kwargs,
    )


def get_emotion_pipeline():
    """Asegura ambos modelos cargados y corre un probe bilingue.

    Lanza ModelBusyError sin VRAM (=> HTTP 503) o RuntimeError si falla
    la carga (sin dummy, sin scores random).
    """
    pipes = {}
    adquiridos = []
    try:
        for lang in ("es", "en"):
            slot = _slot(lang)
            try:
                pipes[lang] = slot.get()
            except ModelBusyError:
                raise
            except Exception as e:
                logger.error(f"[emotion-scoring] no se pudo cargar {lang}: {e}")
                raise RuntimeError(
                    f"No se pudo cargar el modelo de emociones ({lang}).") from e
            adquiridos.append(slot)
        ml_es = _pipe_max_len(pipes["es"])
        ml_en = _pipe_max_len(pipes["en"])
        pipes["es"](["Estoy muy contento hoy."], truncation=True, max_length=ml_es)
        pipes["en"](["I am very happy today."], truncation=True, max_length=ml_en)
    finally:
        for slot in adquiridos:
            try:
                slot.release()
            except Exception:
                pass
    return pipes


def _detect_lang(text):
    try:
        import langid
        lang, _ = langid.classify(text)
        return str(lang).lower()
    except Exception:
        return "es"


def detect_corpus_language(texts, sample=300):
    """Idioma mayoritario del corpus (fallback para textos muy cortos)."""
    cortos = [t for t in texts if isinstance(t, str) and len(t.strip()) >= SHORT_TEXT_CHARS]
    if not cortos:
        return "es"
    muestra = cortos[::max(1, len(cortos) // sample)][:sample]
    votos = {}
    for t in muestra:
        lang = _detect_lang(t)
        votos[lang] = votos.get(lang, 0) + 1
    return max(votos, key=votos.get)


def preprocess_text(text, lang):
    """Limpieza de chat antes de puntuar. Conservadora y por idioma."""
    t = text or ""
    t = _PREFIX_RE.sub("", t)
    t = _URL_RE.sub("", t)
    t = _MENTION_RE.sub("", t)
    t = _HASHTAG_RE.sub(r"\1", t)
    t = _REPEAT_RE.sub(r"\1\1", t)
    if lang == "es":
        try:
            import emoji
            t = emoji.demojize(t, language="es")
        except Exception:
            pass
    return t.strip()


def _windows(pipe, text, max_len):
    """Ventana deslizante por tokens (respeta el tope real del modelo)."""
    try:
        tok = pipe.tokenizer
        ids = tok.encode(text, add_special_tokens=False)
    except Exception:
        return [text[:4000]]
    usable = max(16, max_len - 2)
    if len(ids) <= usable:
        return [text]
    overlap = min(64, usable // 4)
    step = usable - overlap
    partes = []
    for start in range(0, len(ids), step):
        partes.append(tok.decode(ids[start:start + usable], skip_special_tokens=True))
        if len(partes) >= MAX_WINDOWS or start + usable >= len(ids):
            break
    return partes


def _run_pipe(pipe, texts, max_len=512):
    out = pipe(texts, truncation=True, max_length=max_len, batch_size=BATCH_SIZE)
    if isinstance(out, dict):
        return [out]
    return out


def _pipe_max_len(pipe):
    """Tope real de tokens por pasada (robertuito-emotion: 128).

    RoBERTa numera posiciones desde 1 (reserva el 0 para padding), asi que
    el limite seguro es model_max_length del tokenizer, NO
    max_position_embeddings (130) ni 512: ambos rompen este modelo.
    """
    candidatos = [
        getattr(getattr(pipe, "tokenizer", None), "model_max_length", None),
    ]
    try:
        mpe = int(pipe.model.config.max_position_embeddings)
        candidatos.append(mpe - 2)
    except Exception:
        pass
    for cand in candidatos:
        try:
            n = int(cand)
        except (TypeError, ValueError):
            continue
        if 32 <= n <= 512:
            return n
    return 128


def score_texts(texts, fallback_lang="es"):
    """Puntua una lista de textos.

    Devuelve (scores (n,6) float32, meta dict con dominant/confidence/
    neutral/lang por texto). Lanza ModelBusyError sin VRAM (=> HTTP 503)
    o RuntimeError si los modelos fallan.
    """
    for slot in list(_slots.values()):
        slot.maybe_evict()
    pipes = {}
    adquiridos = []
    try:
        for lang in ("es", "en"):
            slot = _slot(lang)
            pipes[lang] = slot.get()
            adquiridos.append(slot)
        return _score_inner(texts, pipes, fallback_lang)
    finally:
        for slot in adquiridos:
            try:
                slot.release()
            except Exception:
                pass


def _score_inner(texts, pipes, fallback_lang="es"):
    corpus_lang = detect_corpus_language(texts)
    corpus_default = corpus_lang if corpus_lang in ("es", "en") else fallback_lang

    langs = []
    limpios = []
    for t in texts:
        t = t if isinstance(t, str) else ""
        if len(t.strip()) < SHORT_TEXT_CHARS:
            lang = corpus_default
        else:
            lang = _detect_lang(t)
            if lang not in ("es", "en"):
                lang = corpus_default
        langs.append(lang)
        limpios.append(preprocess_text(t, lang))

    n = len(texts)
    m = len(EMOTIONS)
    scores = np.zeros((n, m), dtype=np.float32)
    neutral = np.zeros(n, dtype=np.float32)
    dominant = ["neutral"] * n
    confidence = np.zeros(n, dtype=np.float32)

    def _aplicar(pos, res):
        if isinstance(res, dict):
            res = [res]
        bolsa = {}
        for item in res:
            lab = str(item.get("label", "")).lower().strip()
            if lab in TARGETS:
                bolsa[lab] = float(item.get("score", 0.0))
        if not bolsa:
            return
        dom = max(bolsa, key=bolsa.get)
        dominant[pos] = dom
        confidence[pos] = bolsa[dom]
        neutral[pos] = bolsa.get(NEUTRAL, 0.0)
        for j, emo in enumerate(EMOTIONS):
            scores[pos, j] = bolsa.get(emo, 0.0)

    for lang in ("es", "en"):
        idxs = [i for i, l in enumerate(langs) if l == lang and limpios[i]]
        if not idxs:
            continue
        pipe = pipes[lang]
        ml = _pipe_max_len(pipe)
        cortos = [i for i in idxs if len(limpios[i].split()) <= MAX_WORDS_WINDOW]
        largos = [i for i in idxs if len(limpios[i].split()) > MAX_WORDS_WINDOW]
        if cortos:
            lote = [limpios[i] for i in cortos]
            try:
                resultados = _run_pipe(pipe, lote, ml)
            except Exception as e:
                logger.error(f"[emotion-scoring] fallo inferencia ({lang}): {e}")
                raise RuntimeError(f"Fallo la inferencia de emociones ({lang}).") from e
            for pos, res in zip(cortos, resultados):
                _aplicar(pos, res)
        for pos in largos:
            try:
                _aplicar(pos, _score_con_ventana(pipe, limpios[pos], ml))
            except Exception as e:
                logger.error(f"[emotion-scoring] fallo inferencia ventana ({lang}): {e}")
                raise RuntimeError(f"Fallo la inferencia de emociones ({lang}).") from e

    for i in range(n):
        if not limpios[i]:
            dominant[i] = NEUTRAL
            confidence[i] = 1.0
            neutral[i] = 1.0

    meta = {
        "dominant": dominant,
        "confidence": confidence,
        "neutral": neutral,
        "lang": langs,
    }
    return scores, meta


def _score_con_ventana(pipe, text, max_len):
    """Promedia ventanas para un texto largo con un pipe dado."""
    partes = _windows(pipe, text, max_len)
    if len(partes) == 1:
        return _run_pipe(pipe, partes, max_len)[0]
    salidas = _run_pipe(pipe, partes, max_len)
    acumulado: dict = {}
    for res in salidas:
        if isinstance(res, dict):
            res = [res]
        for item in res:
            lab = str(item.get("label", "")).lower().strip()
            acumulado[lab] = acumulado.get(lab, 0.0) + float(item.get("score", 0.0))
    total = float(len(salidas)) or 1.0
    return [{"label": k, "score": v / total} for k, v in acumulado.items()]


def enrich_dataframe_with_emotions(
    df,
    categories=None,
    lexicon=None,
    text_column="texto",
):
    """Agrega columnas de emociones al dataframe (mismo contrato que antes).

    Columnas extra: _neutral, _dominant, _conf, _lang (usadas para calidad).
    Lanza RuntimeError si los modelos fallan (sin dummy, sin ceros silentes).
    """
    import pandas as pd

    if categories is None:
        categories = EMOTIONS
    if text_column not in df.columns:
        raise ValueError(f"Columna '{text_column}' no encontrada. Columnas: {list(df.columns)}")

    enriched = df.copy()
    for cat in categories:
        if cat not in enriched.columns:
            enriched[cat] = 0.0

    texts = enriched[text_column].astype(str).fillna("").tolist()
    texts = [t[:2000] for t in texts]

    scores, meta = score_texts(texts)

    cat_index = {c: i for i, c in enumerate(EMOTIONS)}
    for k, cat in enumerate(categories):
        if cat in cat_index:
            enriched[cat] = scores[:, cat_index[cat]]
    for cat in categories:
        enriched[cat] = pd.to_numeric(enriched[cat], errors="coerce").fillna(0.0).astype(float)

    enriched["_neutral"] = meta["neutral"].astype(float)
    enriched["_dominant"] = meta["dominant"]
    enriched["_conf"] = meta["confidence"].astype(float)
    enriched["_lang"] = meta["lang"]
    return enriched


def predecir_emociones_textos(textos, categories=None):
    """Matriz (n, m) de puntajes 0-1 (mismo contrato que antes)."""
    if categories is None:
        categories = EMOTIONS
    n = len(textos)
    m = len(categories)
    if n == 0:
        return np.zeros((0, m), dtype=np.float32)
    truncados = [(t or "")[:2000] for t in textos]
    scores, _meta = score_texts(truncados)
    cat_index = {c: i for i, c in enumerate(EMOTIONS)}
    salida = np.zeros((n, m), dtype=np.float32)
    for k, cat in enumerate(categories):
        if cat in cat_index:
            salida[:, k] = scores[:, cat_index[cat]]
    return salida


def promediar_emociones(scores, categories=None):
    """Promedia una matriz de puntajes por emocion (mismo contrato)."""
    if categories is None:
        categories = EMOTIONS
    scores = np.asarray(scores, dtype=np.float32)
    if scores.size == 0:
        return {emo: 0.0 for emo in categories}
    if scores.ndim == 1:
        scores = scores.reshape(1, -1)
    m = min(scores.shape[1], len(categories))
    return {emo: float(scores[:, j].mean()) for j, emo in enumerate(categories[:m])}


def quality_summary(df_enriched):
    """Metricas de calidad/calidez del analisis para mostrar en el front."""
    import pandas as pd

    n = int(len(df_enriched))
    if n == 0:
        return {"n_textos": 0, "neutral_rate": 0.0, "low_confidence_rate": 0.0,
                "dominant_counts": {}, "n_es": 0, "n_en": 0}
    dom = df_enriched["_dominant"].astype(str) if "_dominant" in df_enriched.columns else pd.Series([], dtype=str)
    conf = pd.to_numeric(df_enriched["_conf"], errors="coerce").fillna(0.0) if "_conf" in df_enriched.columns else pd.Series([1.0] * n)
    lang = df_enriched["_lang"].astype(str) if "_lang" in df_enriched.columns else pd.Series(["es"] * n)
    counts = dom.value_counts().to_dict() if len(dom) else {}
    return {
        "n_textos": n,
        "neutral_rate": round(float((dom == NEUTRAL).mean()) if len(dom) else 0.0, 4),
        "low_confidence_rate": round(float((conf < LOW_CONF_THRESHOLD).mean()), 4),
        "dominant_counts": {str(k): int(v) for k, v in counts.items()},
        "n_es": int((lang == "es").sum()),
        "n_en": int((lang == "en").sum()),
    }
