import asyncio
import time
import uuid
from typing import Annotated
from fastapi.middleware.cors import CORSMiddleware

from fastapi import (
    Request,
    BackgroundTasks,
    Body,
    FastAPI,
    Header,
    HTTPException,
)

from fastapi.encoders import jsonable_encoder

from .config import get_settings
from .topic_modeling_service import TopicModelingService
from .supabase_service import SupabaseService
from .chroma_service import ChromaService
from .title_provider import TitleProvider
from .topic_modeling_provider import TopicModelingProvider
import logging
from logging_utils import setup_service_logger
import jobs
from .model_cache import ModelBusyError, check_fits_or_raise
logger = setup_service_logger("topicmodeling-service")


settings = get_settings()
app = FastAPI()

origins = ["http://localhost:8080"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

supabase_service = SupabaseService()
chroma_service = ChromaService()
topic_provider = TopicModelingProvider()
title_provider = TitleProvider(
    provider=settings.llm_provider,
    groq_api_key=settings.groq_api_key,
    groq_model=settings.groq_model,
    api_key=settings.google_api_key,
    model=settings.llm_model,
    ollama_base_url=settings.ollama_base_url,
    ollama_model=settings.ollama_model,
    ollama_keep_alive=settings.ollama_keep_alive,
    ollama_timeout_seconds=settings.ollama_timeout_seconds,
    ollama_num_ctx=settings.ollama_num_ctx,
    enabled=settings.title_enabled,
    batch_size=settings.title_batch_size,
    max_chars_per_doc=settings.title_max_chars_per_doc,
)

topic_modeling_service = TopicModelingService(
    supabase_repository=supabase_service,
    chroma_repository=chroma_service,
    topic_provider=topic_provider,
    title_provider=title_provider,
)

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


@app.post("/getTopics", status_code=202)
async def start_get_topics(
    background_tasks: BackgroundTasks,
    dataset: str = Body(..., embed=True),
    max_topics: int = Body(30, embed=True),
    privacy_mode: bool = Body(False, embed=True),
    x_user_id: str = Header(...),
    x_user_token: str = Header(...),
):
    """Encola el modelado de topicos. Devuelve 202 {job_id}; el resultado se
    consulta con GET /jobs/{job_id} (el JSON es el mismo de antes)."""
    supabase = supabase_service.get_supabase_client(x_user_token)
    try:
        job = jobs.create_job(
            supabase, service="topicmodeling", job_type="topics",
            user_id=x_user_id, dataset_name=dataset,
            params={"max_topics": max_topics, "privacy_mode": bool(privacy_mode)},
        )
    except Exception as e:
        logger.exception(f"[topicmodeling] fallo creando job | user_id={x_user_id} causa={type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor.") from e
    background_tasks.add_task(
        _run_topics_job,
        job["job_id"],
        {"dataset": dataset, "max_topics": max_topics, "privacy_mode": bool(privacy_mode)},
        x_user_id,
        x_user_token,
    )
    return {"job_id": job["job_id"], "status": "queued", "dataset": dataset}


def _run_topics_job(job_id: str, payload: dict, user_id: str, user_token: str):
    asyncio.run(_run_topics_job_async(job_id, payload, user_id, user_token))


async def _run_topics_job_async(job_id: str, payload: dict, user_id: str, user_token: str):
    supabase = supabase_service.get_supabase_client(user_token)
    jobs.mark_running(supabase, job_id)
    try:
        result = await _compute_get_topics(
            dataset=payload["dataset"],
            max_topics=payload.get("max_topics", 30),
            privacy_mode=bool(payload.get("privacy_mode", False)),
            x_user_id=user_id, x_user_token=user_token,
        )
    except HTTPException as e:
        logger.warning(f"[topicmodeling] job {job_id} fallo | code={e.status_code} detail={e.detail}")
        jobs.mark_failed(supabase, job_id, e.status_code, str(e.detail))
        return
    except Exception as e:
        logger.exception(f"[topicmodeling] job {job_id} fallo no controlado | causa={type(e).__name__}: {e}")
        jobs.mark_failed(supabase, job_id, 500, "Error interno del servidor.")
        return
    jobs.mark_succeeded(supabase, job_id, jsonable_encoder(result))


@app.get("/jobs/{job_id}")
async def get_job_status(
    job_id: str,
    x_user_id: str = Header(...),
    x_user_token: str = Header(...),
):
    row = jobs.get_job(supabase_service.get_supabase_client(x_user_token), job_id, x_user_id)
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
    return {"jobs": jobs.list_jobs(supabase_service.get_supabase_client(x_user_token), x_user_id, dataset_name, job_type, active_only=True)}


async def _compute_get_topics(
    dataset: str = Body(..., embed=True),
    max_topics: int = Body(30, embed=True),
    privacy_mode: bool = Body(False, embed=True),
    x_user_id: str = Header(...),
    x_user_token: str = Header(...),
):
    try:
        check_fits_or_raise("topicmodeling", 2000)
    except ModelBusyError as e:
        logger.warning(f"[topicmodeling] sin VRAM: {e}")
        raise HTTPException(status_code=503, detail="Servicio ocupado, intenta más tarde.") from e
    try:
        resultado = await topic_modeling_service.get_topics(
            dataset_name=dataset,
            user_id=x_user_id,
            user_token=x_user_token,
            max_topics=max_topics,
            privacy_mode=privacy_mode,
        )

        return {
            "status": "success",
            **resultado,
        }

    except ValueError as exc:
        logger.exception(f"[VALUE ERROR] {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except PermissionError as exc:
        raise HTTPException(
            status_code=403,
            detail=str(exc),
        ) from exc

    except LookupError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception(
            "[topicmodeling] fallo en POST /getTopics | "
            f"method=POST user_id={x_user_id} dataset={dataset} causa={type(exc).__name__}: {exc}"
        )
        raise HTTPException(
            status_code=500,
            detail="Error interno procesando el topic modeling.",
        ) from exc

@app.post("/getTopicResults")
async def get_topic_results(
    dataset: str = Body(..., embed=True),
    max_topics: int = Body(30, embed=True),

    x_user_id: str = Header(...),
    x_user_token: str = Header(...),
):
    """Solo lee el caché de Supabase. No calcula tópicos."""
    try:
        resultado = await topic_modeling_service.get_cached_topics(
            dataset_name=dataset,
            user_id=x_user_id,
            user_token=x_user_token,
            max_topics=max_topics,
        )

        if resultado is None:
            raise HTTPException(
                status_code=404,
                detail="No hay resultados de tópicos para este dataset.",
            )

        return {
            "status": "success",
            **resultado,
        }

    except LookupError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "[topicmodeling] fallo en POST /getTopicResults | "
            f"method=POST user_id={x_user_id} dataset={dataset} causa={type(exc).__name__}: {exc}"
        )
        raise HTTPException(
            status_code=500,
            detail="Error interno obteniendo los tópicos.",
        ) from exc
