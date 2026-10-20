import asyncio
import time
import uuid
from typing import Annotated

from fastapi import (
    Request,
    BackgroundTasks,
    FastAPI,
    Header,
    HTTPException,
)

from fastapi.encoders import jsonable_encoder

import jobs

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from config import get_settings

from chroma_dataset_service import (
    ChromaDatasetService,
)

from download.download_service import (
    DownloadService,
)

from cleaning.cleaning_service import (
    CleaningService,
)

from cleaning.cleaning_types import (
    DownloadRequest,
    CleaningRequest,
)

from supabase_service import (

    SupabaseService,
)
import logging
from logging_utils import setup_service_logger
logger = setup_service_logger("cleaning-service")

settings = get_settings()

app = FastAPI()

origins = [
    "http://localhost:8080"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

dataset_service = ChromaDatasetService()

download_service = DownloadService(
    storage_path=settings.cleaning_download_path
)

supabase_service = SupabaseService()

cleaning_service = CleaningService(
    dataset_service=dataset_service,
    download_service=download_service,
    supabase_service=supabase_service,
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


@app.post("/cleanDataset", status_code=202)
async def start_cleaning(
    request: CleaningRequest,
    background_tasks: BackgroundTasks,
    x_user_id: str = Header(...),
    x_user_token: str = Header(...),
):
    """Encola la limpieza. Devuelve 202 {job_id}; el resultado se consulta
    con GET /jobs/{job_id} (el JSON es el mismo de antes)."""
    supabase = supabase_service.get_supabase_client(x_user_token)
    try:
        job = jobs.create_job(
            supabase, service="datacleaning", job_type="clean",
            user_id=x_user_id, dataset_name=request.dataset_name,
            params=request.model_dump(mode="json"),
        )
    except Exception as e:
        logger.exception(f"[cleaning] fallo creando job | user_id={x_user_id} causa={type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor.") from e
    background_tasks.add_task(_run_clean_job, job["job_id"], request.model_dump(mode="json"), x_user_id, x_user_token)
    return {"job_id": job["job_id"], "status": "queued", "dataset": request.dataset_name}


def _run_clean_job(job_id: str, payload: dict, user_id: str, user_token: str):
    asyncio.run(_run_clean_job_async(job_id, payload, user_id, user_token))


async def _run_clean_job_async(job_id: str, payload: dict, user_id: str, user_token: str):
    supabase = supabase_service.get_supabase_client(user_token)
    jobs.mark_running(supabase, job_id)
    try:
        result = await _compute_cleaning(CleaningRequest(**payload), user_id, user_token)
    except HTTPException as e:
        logger.warning(f"[cleaning] job {job_id} fallo | code={e.status_code} detail={e.detail}")
        jobs.mark_failed(supabase, job_id, e.status_code, str(e.detail))
        return
    except Exception as e:
        logger.exception(f"[cleaning] job {job_id} fallo no controlado | causa={type(e).__name__}: {e}")
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


async def _compute_cleaning(
    request: CleaningRequest,
    x_user_id: str,
    x_user_token: str,
):
    try:

        result = await cleaning_service.clean(
            request=request,
            user_id=x_user_id,
            user_token=x_user_token,
        )

        return {
            "status": "success",
            **result.model_dump(),
        }

    except ValueError as exc:

        logger.exception(f"[CLEANING VALUE ERROR] "

            f"{type(exc).__name__}: {exc}")

        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except PermissionError as exc:

        raise HTTPException(
            status_code=403,
            detail=str(exc),
        ) from exc

    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        logger.exception(f"Error ejecutando cleaning: {exc}")

        raise HTTPException(
            status_code=500,
            detail=(
                "Error interno procesando "
                "la limpieza."
            ),
        ) from exc


@app.post("/download")
async def download_cleaned_dataset(
    request: DownloadRequest,
    x_user_id: str = Header(...),
):
    try:
        download = (
            download_service.get_download(
                download_id=request.download_id,
                user_id=x_user_id,
            )
        )

        return FileResponse(
            path=download.file_path,
            media_type=download.content_type,
            filename=download.filename,
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        logger.exception(f"Error descargando archivo: {exc}")

        raise HTTPException(
            status_code=500,
            detail=(
                "Error descargando "
                "el archivo."
            ),
        ) from exc
