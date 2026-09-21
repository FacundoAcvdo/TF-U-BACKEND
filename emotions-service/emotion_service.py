import pandas as pd
import numpy as np
import re

# 6 emociones multilingüe es/en
EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

# Un solo modelo multilingüe — -50% VRAM/tiempo (antes 2 modelos)
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
    print(f"[emotions-service] device={'GPU' if device==0 else 'CPU'} single-model {MODEL_SINGLE}", flush=True)
    try:
        from transformers import pipeline
        import torch
        # FP16 en GPU para 1.4x speedup, sin cambio de accuracy relevante
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
        # torch.compile deshabilitado: requiere gcc/triton y CC env, en runtime nvidia/cuda no está
        # para 1.4x pro, instalar build-essential en Dockerfile y descomentar abajo
        # if device == 0 and hasattr(torch, "compile"):
        #     try:
        #         pipe.model = torch.compile(pipe.model)  # type: ignore
        #         print("[emotions-service] torch.compile OK (FP16)", flush=True)
        #     except Exception as e:
        #         print(f"[emotions-service] torch.compile skip: {e}", flush=True)
        print("[emotions-service] FP16 OK, compile skipped (necesita gcc)", flush=True)
        _pipeline = pipe
        print(f"[emotions-service] {MODEL_SINGLE} OK", flush=True)
        return _pipeline
    except Exception as e:
        print(f"[emotions-service] fallo carga {MODEL_SINGLE}: {e} -> dummy", flush=True)
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
    # truncado a 2000 chars para evitar OOM con 512 tokens
    texts = [t[:2000] for t in texts]

    if pipe is not None:
        try:
            # Batch vectorizado — GPU 8GB ok con 32 x 512 tokens
            results = pipe(texts, truncation=True, max_length=512, batch_size=32)
            # results: lista de listas [{label,score}]
            n = len(texts)
            m = len(categories)
            scores_array = np.zeros((n, m), dtype=np.float32)
            cat_index = {c: i for i, c in enumerate(categories)}
            for i, res in enumerate(results):
                # res es lista de dicts
                for item in res:
                    lab = str(item.get("label", "")).lower().strip()
                    if lab in cat_index:
                        scores_array[i, cat_index[lab]] = float(item.get("score", 0.0))
            # Vectorizado: una sola asignación en vez de .at en loop
            enriched.loc[:, categories] = scores_array
            # asegura float
            for cat in categories:
                enriched[cat] = pd.to_numeric(enriched[cat], errors="coerce").fillna(0.0).astype(float)
            return enriched
        except Exception as e:
            print(f"[emotions-service] batch vectorizado fallo: {e} -> fallback fila", flush=True)
            # fallback por fila vectorizado parcial
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
        # dummy sin modelo (tests sin internet) — no usa stopwords, solo para no romper pipeline
        print("[emotions-service] dummy scores (sin modelo)", flush=True)
        import random
        n = len(texts)
        m = len(categories)
        dummy = np.random.rand(n, m).astype(np.float32)
        # normaliza por fila
        row_sums = dummy.sum(axis=1, keepdims=True)
        dummy = dummy / np.where(row_sums==0, 1, row_sums)
        enriched.loc[:, categories] = dummy
        return enriched
