import time
import uuid
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
import asyncio
import os
import re
import pandas as pd
import numpy as np
from collections import Counter, defaultdict

from utils import build_dataframe_from_chroma
from chroma_reader import fetch_paginated
from emotion_service import EMOTIONS, enrich_dataframe_with_emotions, get_emotion_pipeline, quality_summary
from stats_service import get_corpus_emotion_stats, get_class_emotion_stats
from supabase_service import (

    get_supabase_client,
    get_dataset_info,
    save_emotions_results,
    get_emotions_results,
)
import logging
from logging_utils import setup_service_logger
import jobs
logger = setup_service_logger("emotions-service")


_FALLBACK_EN = {
    "a","about","above","after","again","against","ain","all","am","an","and","any","are","aren","aren't","as","at","be","because","been","before","being","below","between","both","but","by","can","couldn","couldn't","d","did","didn","didn't","do","does","doesn","doesn't","doing","don","don't","down","during","each","few","for","from","further","had","hadn","hadn't","has","hasn","hasn't","have","haven","haven't","having","he","he'd","he'll","he's","her","here","hers","herself","him","himself","his","how","i","i'd","i'll","i'm","i've","if","in","into","is","isn","isn't","it","it'd","it'll","it's","its","itself","just","ll","m","ma","me","mightn","mightn't","more","most","mustn","mustn't","my","myself","needn","needn't","no","nor","not","now","o","of","off","on","once","only","or","other","our","ours","ourselves","out","over","own","re","s","same","shan","shan't","she","she'd","she'll","she's","should","should've","shouldn","shouldn't","so","some","such","t","than","that","that'll","the","their","theirs","them","themselves","then","there","these","they","they'd","they'll","they're","they've","this","those","through","to","too","under","until","up","ve","very","was","wasn","wasn't","we","we'd","we'll","we're","we've","were","weren","weren't","what","when","where","which","while","who","whom","why","will","with","won","won't","wouldn","wouldn't","y","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves"
}
_FALLBACK_ES = {
    "de","la","que","el","en","y","a","los","del","se","las","por","un","para","con","no","una","su","al","lo","como","más","pero","sus","le","ya","o","este","sí","porque","esta","entre","cuando","muy","sin","sobre","también","me","hasta","hay","donde","quien","desde","todo","nos","durante","todos","uno","les","ni","contra","otros","ese","eso","ante","ellos","e","esto","mí","antes","algunos","qué","unos","yo","otro","otras","otra","él","tanto","esa","estos","mucho","quienes","nada","muchos","cual","poco","ella","estar","estas","algunas","algo","nosotros","mi","mis","tú","te","ti","tu","tus","ellas","nosotras","vosotros","vosotras","os","mío","mía","míos","mías","tuyo","tuya","tuyos","tuyas","suyo","suya","suyos","suyas","nuestro","nuestra","nuestros","nuestras","vuestro","vuestra","vuestros","vuestras","esos","esas","estoy","estás","está","estamos","estáis","están","esté","estés","estemos","estéis","estén","estaré","estarás","estará","estaremos","estaréis","estarán","estaría","estarías","estaríamos","estaríais","estarían","estaba","estabas","estábamos","estabais","estaban","estuve","estuviste","estuvo","estuvimos","estuvisteis","estuvieron","estuviera","estuvieras","estuviéramos","estuvierais","estuvieran","estuviese","estuvieses","estuviésemos","estuvieseis","estuviesen","estando","estado","estada","estados","estadas","estad","he","has","ha","hemos","habéis","han","haya","hayas","hayamos","hayáis","hayan","habré","habrás","habrá","habremos","habréis","habrán","habría","habrías","habríamos","habríais","habrían","había","habías","habíamos","habíais","habían","hube","hubiste","hubo","hubimos","hubisteis","hubieron","hubiera","hubieras","hubiéramos","hubierais","hubieran","hubiese","hubieses","hubiésemos","hubieseis","hubiesen","habiendo","habido","habida","habidos","habidas","soy","eres","es","somos","sois","son","sea","seas","seamos","seáis","sean","seré","serás","será","seremos","seréis","serán","sería","serías","seríamos","seríais","serían","era","eras","éramos","erais","eran","fui","fuiste","fue","fuimos","fuisteis","fueron","fuera","fueras","fuéramos","fuerais","fueran","fuese","fueses","fuésemos","fueseis","fuesen","sintiendo","sentido","sentida","sentidos","sentidas","siente","sentid","tengo","tienes","tiene","tenemos","tenéis","tienen","tenga","tengas","tengamos","tengáis","tengan","tendré","tendrás","tendrá","tendremos","tendréis","tendrán","tendría","tendrías","tendríamos","tendríais","tendrían","tenía","tenías","teníamos","teníais","tenían","tuve","tuviste","tuvo","tuvimos","tuvisteis","tuvieron","tuviera","tuvieras","tuviéramos","tuvierais","tuvieran","tuviese","tuvieses","tuviésemos","tuvieseis","tuviesen","teniendo","tenido","tenida","tenidos","tenidas","tened"
}

_FALLBACK_ES_NORM = {w for w in _FALLBACK_ES}
for _w in list(_FALLBACK_ES):
    _norm = _w.replace("á","a").replace("é","e").replace("í","i").replace("ó","o").replace("ú","u").replace("ü","u")
    _FALLBACK_ES_NORM.add(_norm)
_FALLBACK_ES = _FALLBACK_ES_NORM

try:
    from nltk.corpus import stopwords
    try:
        STOPWORDS = set(stopwords.words("spanish")) | set(stopwords.words("english"))
    except LookupError:
        import nltk
        try:
            nltk.download("stopwords", quiet=True)
            STOPWORDS = set(stopwords.words("spanish")) | set(stopwords.words("english"))
        except Exception as e2:
            logger.warning(f"[emotions-service] nltk download fallo ({e2}) -> fallback hardcodeado")
            STOPWORDS = _FALLBACK_ES | _FALLBACK_EN

    STOPWORDS |= _FALLBACK_ES | _FALLBACK_EN

    STOPWORDS = {w.lower() for w in STOPWORDS}
    pass
except Exception as e:
    logger.warning(f"[emotions-service] NLTK stopwords no disponible ({e}) -> fallback hardcodeado {len(_FALLBACK_ES|_FALLBACK_EN)}")
    STOPWORDS = _FALLBACK_ES | _FALLBACK_EN

@asynccontextmanager
async def lifespan(app: FastAPI):

    pass
    try:

        await asyncio.to_thread(get_emotion_pipeline)
        pass
    except Exception as e:
        logger.exception(f"[emotions-service] warmup fallo (seguirá lazy en primera request): {e}")
    yield

app = FastAPI(title="Emotions Service", lifespan=lifespan)

origins = ["http://localhost:8080"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class EmotionsRequest(BaseModel):
    dataset: str


def _clean_tokens(text: str) -> list[str]:
    text = str(text).lower()
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"[^a-záéíóúüñ\s]", " ", text)
    tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]

def build_correlation_matrix(df, emotions):
    emotions_present = [e for e in emotions if e in df.columns]
    if not emotions_present or df.empty:
        return {"matrix": [], "order": emotions, "flat": []}
    corr = df[emotions_present].corr(method="pearson").fillna(0.0)
    order = df[emotions_present].mean().sort_values(ascending=False).index.tolist()
    corr = corr.reindex(index=order, columns=order)
    matrix = corr.values.tolist()
    flat = []
    for i, row_emo in enumerate(order):
        for j, col_emo in enumerate(order):
            flat.append({"x": col_emo, "y": row_emo, "value": float(matrix[i][j])})
    return {"matrix": matrix, "order": order, "flat": flat}

def build_distribution_data(df, emotions, bins=20):
    result = []
    for emo in emotions:
        if emo not in df.columns or df.empty:
            result.append({"emotion": emo, "stats": {}, "histogram": []})
            continue
        series = df[emo].astype(float)
        q25 = float(series.quantile(0.25))
        q75 = float(series.quantile(0.75))
        median = float(series.median())
        mean = float(series.mean())
        min_v = float(series.min())
        max_v = float(series.max())
        std = float(series.std(ddof=1) if len(series) > 1 else 0.0)
        counts, bin_edges = np.histogram(series, bins=bins, range=(0, 1))
        histogram = []
        for i in range(len(counts)):
            histogram.append({
                "bin_start": float(bin_edges[i]),
                "bin_end": float(bin_edges[i+1]),
                "count": int(counts[i]),
            })
        result.append({
            "emotion": emo,
            "stats": {
                "count": int(series.count()),
                "mean": mean,
                "std": std,
                "min": min_v,
                "max": max_v,
                "median": median,
                "q25": q25,
                "q75": q75,
            },
            "histogram": histogram,
        })
    result.sort(key=lambda x: x["stats"].get("mean", 0), reverse=True)
    return result

def _top_words_weighted(df_subset: "pd.DataFrame", emotions: list[str], text_column: str, top_n: int, min_df: int) -> dict:
    """
    Helper: top palabras por emoción basadas en intensidad del modelo.
    Para cada palabra w, calcula mean_score[w][emo] = promedio de score emo en docs que contienen w.
    Luego rankea por mean_score descendente (con desempate por df).
    Esto asegura que sadness -> palabras que aparecen en docs con alto score sadness,
    no simplemente las más frecuentes en docs con sadness dominante.
    """
    top_words = {emo: [] for emo in emotions}
    if df_subset.empty or text_column not in df_subset.columns:
        return top_words

    n = len(df_subset)

    inverted: dict[str, list[int]] = defaultdict(list)

    for idx, txt in enumerate(df_subset[text_column].astype(str)):
        tokens = _clean_tokens(txt)
        if not tokens:
            continue
        uniq = set(tokens)
        for w in uniq:
            inverted[w].append(idx)

    if not inverted:
        return top_words


    emo_scores: dict[str, np.ndarray] = {}
    for emo in emotions:
        if emo in df_subset.columns:
            emo_scores[emo] = pd.to_numeric(df_subset[emo], errors="coerce").fillna(0.0).values.astype(float)
        else:
            emo_scores[emo] = np.zeros(n, dtype=float)


    global_std = max(float(np.std(v)) for v in emo_scores.values()) if emo_scores else 0.0
    use_freq_fallback = global_std < 1e-6

    for emo in emotions:
        scores = emo_scores[emo]

        if use_freq_fallback:

            all_tokens = []
            for txt in df_subset[text_column].astype(str):
                all_tokens.extend(_clean_tokens(txt))
            cnt = Counter(all_tokens)
            top_words[emo] = [w for w, _ in cnt.most_common(top_n)]
            continue


        candidates: list[tuple[str, float, int]] = []
        for w, idxs in inverted.items():
            df_w = len(idxs)
            if df_w < min_df:
                continue
            mean_score = float(scores[idxs].mean()) if idxs else 0.0
            candidates.append((w, mean_score, df_w))


        if len(candidates) < top_n:
            for fallback_min in (2, 1):
                if fallback_min >= min_df:
                    continue
                for w, idxs in inverted.items():
                    df_w = len(idxs)
                    if df_w < fallback_min or df_w >= min_df:
                        continue

                    if any(c[0] == w for c in candidates):
                        continue
                    mean_score = float(scores[idxs].mean()) if idxs else 0.0
                    candidates.append((w, mean_score, df_w))
                if len(candidates) >= top_n:
                    break


        candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)


        top_words[emo] = [w for w, _, _ in candidates[:top_n]]

        if not top_words[emo]:

            all_tokens = []
            for txt in df_subset[text_column].astype(str):
                all_tokens.extend(_clean_tokens(txt))
            cnt = Counter(all_tokens)
            top_words[emo] = [w for w, _ in cnt.most_common(top_n)]


        top_words[emo] = top_words[emo][:top_n]

    return top_words


def build_top_words(df, emotions, text_column="texto", top_n=3):
    """
    Top 3 palabras representativas por emoción, relacionadas con cada emoción.
    Criterio nuevo: para cada emoción, se promedia el score del modelo en los docs
    que contienen cada palabra; las 3 palabras con mayor promedio son las más
    asociadas a esa emoción (no las más frecuentes globalmente).
    Reemplaza el viejo argmax por doc (que daba palabras genéricas como dont/know).
    """
    if df.empty or text_column not in df.columns:
        return {emo: [] for emo in emotions}
    n = len(df)

    if n < 50:
        min_df = 2
    elif n < 200:
        min_df = 3
    elif n < 500:
        min_df = 5
    else:
        min_df = max(5, n // 100)
    return _top_words_weighted(df, emotions, text_column, top_n, min_df)


def build_class_top_words(df, emotions, text_column="texto", class_column="clase", top_n=3):
    """
    Top 3 por clase y emoción: {clase: {emotion: [w1,w2,w3]}}
    Aplica el mismo criterio ponderado por score a cada subset de clase.
    """
    result = {}
    if df.empty or class_column not in df.columns:
        return result
    for clase in df[class_column].dropna().unique():
        clase_str = str(clase)
        subset = df[df[class_column] == clase]
        n_sub = len(subset)
        if n_sub < 20:
            min_df = 2
        elif n_sub < 100:
            min_df = 3
        else:
            min_df = max(3, n_sub // 50)
        per_emo = _top_words_weighted(subset, emotions, text_column, top_n, min_df)


        for emo in emotions:
            if not per_emo.get(emo):
                all_tokens = []
                for _, r in subset.iterrows():
                    all_tokens.extend(_clean_tokens(r.get(text_column, "")))
                per_emo[emo] = [w for w, _ in Counter(all_tokens).most_common(top_n)]
        result[clase_str] = per_emo
    return result

def enrich_class_stats_with_top_words(class_stats, class_top_words):
    for entry in class_stats:
        clase = str(entry.get("clase", ""))
        per_emo = class_top_words.get(clase, {})
        for emo in EMOTIONS:
            if emo in entry and isinstance(entry[emo], dict):
                entry[emo]["top_words"] = per_emo.get(emo, [])

@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    inicio = time.perf_counter()
    try:
        response = await call_next(request)
        latencia_ms = int((time.perf_counter() - inicio) * 1000)
        user_id = request.headers.get("x-user-id", "-")
        logger.info(
            f"{request.method} {request.url.path} "
            f"status={response.status_code} latencia_ms={latencia_ms} "
            f"req={request_id} user_id={user_id}"
        )
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception as e:
        latencia_ms = int((time.perf_counter() - inicio) * 1000)
        logger.exception(
            f"fallo no controlado {request.method} {request.url.path} "
            f"req={request_id} latencia_ms={latencia_ms} causa={type(e).__name__}: {e}"
        )
        raise


@app.post("/analyzeEmotions", status_code=202)
async def start_analyze_emotions(
    data: EmotionsRequest,
    background_tasks: BackgroundTasks,
    x_user_id: str = Header(...),
    x_user_token: str = Header(...),
):
    """Encola el analisis de emociones. Devuelve 202 {job_id}; el resultado
    se consulta con GET /jobs/{job_id} (el JSON es el mismo de antes)."""
    supabase = get_supabase_client(x_user_token)
    try:
        job = jobs.create_job(
            supabase, service="emotions", job_type="emotions",
            user_id=x_user_id, dataset_name=data.dataset, params={},
        )
    except Exception as e:
        logger.exception(f"[emotions] fallo creando job | user_id={x_user_id} causa={type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor.") from e
    background_tasks.add_task(_run_emotions_job, job["job_id"], data.model_dump(mode="json"), x_user_id, x_user_token)
    return {"job_id": job["job_id"], "status": "queued", "dataset": data.dataset}


async def _run_emotions_job(job_id: str, payload: dict, user_id: str, user_token: str):
    supabase = get_supabase_client(user_token)
    jobs.mark_running(supabase, job_id)
    try:
        result = await _compute_analyze_emotions(EmotionsRequest(**payload), user_id, user_token)
    except HTTPException as e:
        logger.warning(f"[emotions] job {job_id} fallo | code={e.status_code} detail={e.detail}")
        jobs.mark_failed(supabase, job_id, e.status_code, str(e.detail))
        return
    except Exception as e:
        logger.exception(f"[emotions] job {job_id} fallo no controlado | causa={type(e).__name__}: {e}")
        jobs.mark_failed(supabase, job_id, 500, "Error interno del servidor.")
        return
    jobs.mark_succeeded(supabase, job_id, jsonable_encoder(result))


@app.get("/jobs/{job_id}")
async def get_job_status(
    job_id: str,
    x_user_id: str = Header(...),
    x_user_token: str = Header(...),
):
    supabase = get_supabase_client(x_user_token)
    row = jobs.get_job(supabase, job_id, x_user_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Job no encontrado.")
    return row


@app.get("/jobs")
async def list_active_jobs(
    dataset_name: str | None = None,
    job_type: str | None = None,
    x_user_id: str = Header(...),
    x_user_token: str = Header(...),
):
    """Trabajos activos del usuario (para retomar al volver a la pantalla)."""
    supabase = get_supabase_client(x_user_token)
    return {"jobs": jobs.list_jobs(supabase, x_user_id, dataset_name, job_type, active_only=True)}


async def _compute_analyze_emotions(
    data: EmotionsRequest,
    x_user_id: str,
    x_user_token: str,
):
    try:
        supabase = get_supabase_client(x_user_token)
        dataset_info = get_dataset_info(supabase, data.dataset, x_user_id)
        dataset_id = dataset_info["dataset_id"]

        chroma_data = await fetch_paginated(
            {"$and": [{"user_id": x_user_id}, {"dataset_id": dataset_id}]},
            ["documents", "metadatas"],
        )
        if not chroma_data.get("ids"):
            raise HTTPException(status_code=404, detail="No documents found for dataset in Chroma")
        df = build_dataframe_from_chroma(
            chroma_data.get("documents", []),
            chroma_data.get("metadatas", []),
        )
        if df.empty:
            raise HTTPException(status_code=400, detail="Dataset is empty")

        df_enriched = enrich_dataframe_with_emotions(df, categories=EMOTIONS)

        corpus_stats = get_corpus_emotion_stats(df_enriched, EMOTIONS)
        class_stats = get_class_emotion_stats(df_enriched, EMOTIONS)
        correlation = build_correlation_matrix(df_enriched, EMOTIONS)
        distribution = build_distribution_data(df_enriched, EMOTIONS, bins=20)
        top_words = build_top_words(df_enriched, EMOTIONS, text_column="texto", top_n=3)
        class_top_words = build_class_top_words(df_enriched, EMOTIONS, text_column="texto", class_column="clase", top_n=3)
        enrich_class_stats_with_top_words(class_stats, class_top_words)
        calidad = quality_summary(df_enriched)

        try:
            save_emotions_results(supabase, x_user_id, dataset_id, EMOTIONS, corpus_stats, class_stats, correlation, distribution, top_words, class_top_words, calidad)
        except Exception as e:
            logger.warning(f"[emotions-service] warn save_emotions_results: {e}")

        return {
            "dataset_id": dataset_id,
            "emotions": EMOTIONS,
            "corpus_stats": corpus_stats,
            "class_stats": class_stats,
            "correlation": correlation,
            "distribution": distribution,
            "top_words": top_words,
            "class_top_words": class_top_words,
            "calidad": calidad,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[emotions-service] Error al procesar emociones: {str(e)}")
        raise HTTPException(status_code=500, detail="Error interno al procesar emociones.")


@app.post("/getResults")
async def get_results(
    data: EmotionsRequest,
    x_user_id: str = Header(...),
    x_user_token: str = Header(...),
):
    try:
        supabase = get_supabase_client(x_user_token)
        dataset_info = get_dataset_info(supabase, data.dataset, x_user_id)
        dataset_id = dataset_info["dataset_id"]
        data_db = get_emotions_results(supabase, x_user_id, dataset_id)
        return {"dataset_id": dataset_id, **data_db}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[emotions-service] Error al obtener resultados: {str(e)}")
        raise HTTPException(status_code=500, detail="Error interno al obtener resultados.")
