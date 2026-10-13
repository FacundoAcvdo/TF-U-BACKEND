import time
import uuid
from fastapi.middleware.cors import CORSMiddleware
from fastapi import BackgroundTasks, FastAPI, HTTPException, Header, Request
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

import os
import re
from collections import Counter

from nltk.corpus import stopwords

from supabase import create_client
from chroma_reader import fetch_paginated
import logging
from logging_utils import setup_service_logger
import jobs
logger = setup_service_logger("wordcloud-service")


class WordCloudRequest(BaseModel):
    dataset: str

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")

STOPWORDS = (
    set(stopwords.words("spanish"))
    | set(stopwords.words("english"))
)

app = FastAPI()

origins = ["http://localhost:8080"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_supabase_client(user_jwt: str):
    client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    client.postgrest.auth(user_jwt)
    return client

def save_wordcloud_results(supabase, user_id: str, dataset_id: str, frequencies: list, total_words: int, unique_words: int, max_count: int, min_count: int, documents_processed: int):
    payload = {
        "dataset_id": dataset_id,
        "user_id": user_id,
        "frequencies": frequencies,
        "total_words": total_words,
        "unique_words": unique_words,
        "max_count": max_count,
        "min_count": min_count,
        "documents_processed": documents_processed,
    }
    try:
        supabase.table("wordcloud_results").upsert(payload, on_conflict="dataset_id,user_id").execute()
    except Exception:
        try:
            supabase.table("wordcloud_results").delete().eq("dataset_id", dataset_id).eq("user_id", user_id).execute()
        except Exception:
            pass
        supabase.table("wordcloud_results").insert(payload).execute()

def get_wordcloud_results(supabase, user_id: str, dataset_id: str):
    response = (
        supabase.table("wordcloud_results")
        .select("frequencies, total_words, unique_words, max_count, min_count, documents_processed")
        .eq("dataset_id", dataset_id)
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    if not response.data:
        raise HTTPException(status_code=404, detail="No wordcloud results found for this dataset")
    row = response.data[0]
    return {
        "frequencies": row["frequencies"],
        "total_words": row["total_words"],
        "unique_words": row["unique_words"],
        "max_count": row["max_count"],
        "min_count": row["min_count"],
        "documents_processed": row["documents_processed"],
    }

def limpiar_texto(texto: str) -> str:
    texto = texto.lower()
    texto = re.sub(r"\b\d+\b", " ", texto)
    texto = re.sub(r"[^a-záéíóúüñ\s]", " ", texto)
    palabras = texto.split()
    palabras = [
        palabra
        for palabra in palabras
        if palabra not in STOPWORDS
        and len(palabra) > 2
    ]
    return " ".join(palabras)

def build_wordcloud_data(textos_limpios: list[str], max_words: int = 200):

    all_tokens = []

    doc_freq = Counter()
    for t in textos_limpios:
        tokens = t.split()
        all_tokens.extend(tokens)

        doc_freq.update(set(tokens))

    total_words = len(all_tokens)
    counter = Counter(all_tokens)
    unique_words = len(counter)

    most_common = counter.most_common(max_words)
    if not most_common:
        return {
            "total_words": 0,
            "unique_words": 0,
            "frequencies": [],
            "max_count": 0,
            "min_count": 0,
        }

    max_count = most_common[0][1]
    min_count = most_common[-1][1]

    frequencies = []
    for rank, (word, count) in enumerate(most_common, 1):
        percentage = (count / total_words * 100) if total_words else 0
        df = doc_freq.get(word, 0)
        frequencies.append({
            "text": word,
            "value": count,
            "count": count,
            "percentage": round(percentage, 2),
            "doc_frequency": df,
            "doc_percentage": round((df / len(textos_limpios) * 100) if textos_limpios else 0, 1),
            "rank": rank,
        })

    return {
        "total_words": total_words,
        "unique_words": unique_words,
        "frequencies": frequencies,
        "max_count": max_count,
        "min_count": min_count,
    }

async def load_and_process_dataset(x_user_id: str, dataset_id: str):

    where = {
        "$and": [
            {"user_id": x_user_id},
            {"dataset_id": dataset_id},
        ]
    }
    data = await fetch_paginated(where, ["documents"])
    documents = data.get("documents", [])
    if not documents:
        raise HTTPException(status_code=404, detail="No documents found for dataset in Chroma")
    textos = []
    for texto_raw in documents:
        texto = (texto_raw or "").strip()
        if texto:
            textos.append(limpiar_texto(texto))
    texto_completo = " ".join(textos)
    if not texto_completo.strip():
        raise HTTPException(status_code=400, detail="The dataset does not contain usable text.")
    return textos, len(textos)

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


@app.post("/visualizeWordCloud", status_code=202)
async def start_wordcloud(
    data: WordCloudRequest,
    background_tasks: BackgroundTasks,
    x_user_id: str = Header(...),
    x_user_token: str = Header(...),
):
    """Encola la nube de palabras. Devuelve 202 {job_id}; el resultado se
    consulta con GET /jobs/{job_id} (el JSON es el mismo de antes)."""
    supabase = get_supabase_client(x_user_token)
    try:
        job = jobs.create_job(
            supabase, service="wordcloud", job_type="wordcloud",
            user_id=x_user_id, dataset_name=data.dataset, params={},
        )
    except Exception as e:
        logger.exception(f"[wordcloud] fallo creando job | user_id={x_user_id} causa={type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor.") from e
    background_tasks.add_task(_run_wordcloud_job, job["job_id"], data.model_dump(mode="json"), x_user_id, x_user_token)
    return {"job_id": job["job_id"], "status": "queued", "dataset": data.dataset}


async def _run_wordcloud_job(job_id: str, payload: dict, user_id: str, user_token: str):
    supabase = get_supabase_client(user_token)
    jobs.mark_running(supabase, job_id)
    try:
        result = await _compute_wordcloud(WordCloudRequest(**payload), user_id, user_token)
    except HTTPException as e:
        logger.warning(f"[wordcloud] job {job_id} fallo | code={e.status_code} detail={e.detail}")
        jobs.mark_failed(supabase, job_id, e.status_code, str(e.detail))
        return
    except Exception as e:
        logger.exception(f"[wordcloud] job {job_id} fallo no controlado | causa={type(e).__name__}: {e}")
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


async def _compute_wordcloud(
    data: WordCloudRequest,
    x_user_id: str,
    x_user_token: str,
):
    try:
        supabase = get_supabase_client(x_user_token)

        response = (
            supabase
            .table("datasets")
            .select("dataset_id")
            .eq("dataset_name", data.dataset)
            .eq("user_id", x_user_id)
            .execute()
        )

        if not response.data:
            raise HTTPException(status_code=404, detail="Dataset not found or access denied.")

        dataset_info = response.data[0]
        dataset_id = dataset_info.get("dataset_id")
        if not dataset_id:
            raise HTTPException(status_code=500, detail="Dataset does not have a dataset_id.")

        textos_limpios, docs_processed = await load_and_process_dataset(x_user_id, dataset_id)

        wc_data = build_wordcloud_data(textos_limpios, max_words=200)

        try:
            save_wordcloud_results(
                supabase, x_user_id, dataset_id,
                wc_data["frequencies"], wc_data["total_words"], wc_data["unique_words"],
                wc_data["max_count"], wc_data["min_count"], docs_processed
            )
        except Exception as e:
            logger.warning(f"[wordcloud-service] warn save_wordcloud_results: {e}")

        return {
            "dataset_id": dataset_id,
            "documents_processed": docs_processed,
            **wc_data,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error al procesar wordcloud: {str(e)}")
        raise HTTPException(status_code=500, detail="Error interno al procesar wordcloud.")

@app.post("/getWordCloudImage")
async def getWordCloudImage(
    x_user_id: str = Header(...),
    x_user_token: str = Header(...),
    data: WordCloudRequest = None
):
    try:
        supabase = get_supabase_client(x_user_token)
        if data is None or not data.dataset:
            raise HTTPException(status_code=400, detail="Dataset is required")
        response = (
            supabase
            .table("datasets")
            .select("dataset_id")
            .eq("dataset_name", data.dataset)
            .eq("user_id", x_user_id)
            .execute()
        )
        if not response.data:
            raise HTTPException(status_code=404, detail="Dataset not found for the user.")
        dataset_id = response.data[0].get("dataset_id")
        data_db = get_wordcloud_results(supabase, x_user_id, dataset_id)
        return {"dataset_id": dataset_id, **data_db}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error al obtener wordcloud: {str(e)}")
        raise HTTPException(status_code=500, detail="Error interno al obtener wordcloud.")
