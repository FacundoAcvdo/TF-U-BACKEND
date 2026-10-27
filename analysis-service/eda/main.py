import asyncio
import time
import uuid
from fastapi.middleware.cors import CORSMiddleware
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
import numpy as np
from collections import Counter

import jobs

from .stats_service import (
    get_dataset_row_analysis,
    get_estadisticas_corpus,
    get_estadisticas_clases
)

from .supabase_service import (
    get_supabase_client,
    get_dataset_info,
    save_eda_results,
    get_eda_results,
)

from .utils import build_dataframe_from_chroma
from .chroma_reader import fetch_paginated
import logging
from logging_utils import setup_service_logger
logger = setup_service_logger("eda-service")


app = FastAPI()

origins = ["http://localhost:8080"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class EdaRequest(BaseModel):
    dataset: str

def build_sentence_stats(dataset):
    s = dataset["sentences"].astype(float)
    return {
        "count": int(s.count()),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1) if len(s) > 1 else 0.0),
        "min": int(s.min()),
        "max": int(s.max()),
        "median": float(s.median()),
        "q25": float(s.quantile(0.25)),
        "q75": float(s.quantile(0.75)),
    }

def build_sentence_freq(dataset, max_points=40):
    cnt = Counter(dataset["sentences"].tolist())

    items = sorted(cnt.items())

    if len(items) > max_points:

        vals = dataset["sentences"].astype(int).tolist()
        hist, edges = np.histogram(vals, bins=max_points)
        return [{"sentences": int(edges[i]), "count": int(hist[i]), "bin_end": int(edges[i+1])} for i in range(len(hist))]
    return [{"sentences": int(k), "count": int(v)} for k, v in items]

def build_distribution_histograms(dataset, bins=20):
    clases = np.unique(dataset['clase'])
    result = []
    for clase in clases:
        grupo = dataset[dataset['clase'] == clase]
        words = grupo['words'].astype(int).tolist()
        sents = grupo['sentences'].astype(int).tolist()


        if words:
            w_counts, w_edges = np.histogram(words, bins=bins)
            words_hist = [{"bin_start": int(w_edges[i]), "bin_end": int(w_edges[i+1]), "count": int(w_counts[i])} for i in range(len(w_counts))]
        else:
            words_hist = []


        if sents:
            s_counts, s_edges = np.histogram(sents, bins=bins)
            sents_hist = [{"bin_start": int(s_edges[i]), "bin_end": int(s_edges[i+1]), "count": int(s_counts[i])} for i in range(len(s_counts))]
        else:
            sents_hist = []

        result.append({
            "clase": str(clase),
            "words_histogram": words_hist,
            "sentences_histogram": sents_hist,
            "words_stats": {
                "count": len(words),
                "mean": float(np.mean(words)) if words else 0,
                "min": int(np.min(words)) if words else 0,
                "max": int(np.max(words)) if words else 0,
            },
            "sentences_stats": {
                "count": len(sents),
                "mean": float(np.mean(sents)) if sents else 0,
                "min": int(np.min(sents)) if sents else 0,
                "max": int(np.max(sents)) if sents else 0,
            }
        })
    return result

def build_eda_response(dataset):
    dataset = get_dataset_row_analysis(dataset)
    DF_CORPUS = get_estadisticas_corpus(dataset)
    DF_CLASES = get_estadisticas_clases(dataset)

    corpus_stats = DF_CORPUS.to_dict(orient="records")[0]
    class_stats = DF_CLASES.to_dict(orient="records")

    sentence_stats = build_sentence_stats(dataset)
    sentence_freq = build_sentence_freq(dataset)
    distribution = build_distribution_histograms(dataset, bins=20)

    return {
        "corpus_stats": corpus_stats,
        "class_stats": class_stats,
        "sentence_stats": sentence_stats,
        "sentence_freq": sentence_freq,
        "distribution": distribution,
    }

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


@app.post("/eda", status_code=202)
async def start_eda(
    data: EdaRequest,
    background_tasks: BackgroundTasks,
    x_user_id: str = Header(...),
    x_user_token: str = Header(...),
):
    """Encola el analisis EDA. Devuelve 202 {job_id}; el resultado se
    consulta con GET /jobs/{job_id} (el JSON es el mismo de antes)."""
    supabase = get_supabase_client(x_user_token)
    try:
        job = jobs.create_job(
            supabase, service="eda", job_type="eda",
            user_id=x_user_id, dataset_name=data.dataset, params={},
        )
    except Exception as e:
        logger.exception(f"[eda-service] fallo creando job EDA | user_id={x_user_id} causa={type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor.") from e
    background_tasks.add_task(_run_eda_job, job["job_id"], data.model_dump(), x_user_id, x_user_token)
    return {"job_id": job["job_id"], "status": "queued", "dataset": data.dataset}


def _run_eda_job(job_id: str, payload: dict, user_id: str, user_token: str):
    asyncio.run(_run_eda_job_async(job_id, payload, user_id, user_token))


async def _run_eda_job_async(job_id: str, payload: dict, user_id: str, user_token: str):
    supabase = get_supabase_client(user_token)
    jobs.mark_running(supabase, job_id)
    try:
        result = await _compute_eda(EdaRequest(**payload), user_id, user_token)
    except HTTPException as e:
        logger.warning(f"[eda-service] job {job_id} fallo | code={e.status_code} detail={e.detail}")
        jobs.mark_failed(supabase, job_id, e.status_code, str(e.detail))
        return
    except Exception as e:
        logger.exception(f"[eda-service] job {job_id} fallo no controlado | causa={type(e).__name__}: {e}")
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


async def _compute_eda(
    data: EdaRequest,
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

        dataset = build_dataframe_from_chroma(
            chroma_data.get("documents", []),
            chroma_data.get("metadatas", []),
        )
        if dataset.empty:
            raise HTTPException(status_code=400, detail="Dataset is empty")

        clases = dataset["clase"] if "clase" in dataset.columns else None
        sin_clases = (
            clases is None
            or bool(clases.isna().all())
            or bool((clases[~clases.isna()].astype(str).str.strip() == "").all())
        )
        if sin_clases:
            raise HTTPException(
                status_code=422,
                detail="El dataset no tiene clases. Cargá un dataset clasificado para calcular estadísticas.",
            )

        result = build_eda_response(dataset)
        result["dataset_id"] = dataset_id


        try:
            save_eda_results(
                supabase, x_user_id, dataset_id,
                result["corpus_stats"], result["class_stats"],
                result["sentence_stats"], result["sentence_freq"],
                result["distribution"]
            )
        except Exception as e:
            logger.warning(f"[eda-service] warn save_eda_results: {e}")

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(
            "[eda-service] fallo en POST /eda | "
            f"method=POST user_id={x_user_id} dataset={data.dataset} "
            f"causa={type(e).__name__}: {e}"
        )
        raise HTTPException(status_code=500, detail="Error interno del servidor.") from e

@app.post("/getResults")
async def get_results(
    data: EdaRequest,
    x_user_id: str = Header(...),
    x_user_token: str = Header(...),
):
    try:
        supabase = get_supabase_client(x_user_token)
        dataset_info = get_dataset_info(supabase, data.dataset, x_user_id)
        dataset_id = dataset_info["dataset_id"]


        db = get_eda_results(supabase, x_user_id, dataset_id)
        return {"dataset_id": dataset_id, **db}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(
            "[eda-service] fallo en POST /getResults | "
            f"method=POST user_id={x_user_id} dataset={data.dataset} "
            f"causa={type(e).__name__}: {e}"
        )
        raise HTTPException(status_code=500, detail="Error interno del servidor.") from e
