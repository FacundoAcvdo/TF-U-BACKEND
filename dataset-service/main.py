import time
import uuid
import os
import tempfile
from pathlib import Path
from typing import Annotated

from contextlib import asynccontextmanager
import logging
import torch

from fastapi import (
    Request,
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Header,
    UploadFile,
    status,
)
from pydantic import BaseModel, Field
from fastapi.encoders import jsonable_encoder

from chroma_repository import ChromaRepository
from config import get_settings
from dependencies import (
    RequestContext,
    get_request_context,
)
from embeddings import (
    LocalSentenceTransformerProvider,
)
from services import DatasetService
from supabase_repository import (
    SupabaseRepository,
)

from fastapi.middleware.cors import CORSMiddleware

from logging_utils import setup_service_logger
import jobs

settings = get_settings()

logger = setup_service_logger("dataset-service")

@asynccontextmanager
async def lifespan(app: FastAPI):

    try:
        if not torch.cuda.is_available():
            logger.warning("[MODO CPU] CUDA no disponible. Corriendo en procesador.")
    except Exception as exc:
        logger.warning(f"No se pudo verificar GPU: {exc}")

    yield


    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        logger.warning(f"No se pudo liberar memoria GPU al apagar: {exc}")

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan
)

origins = os.environ.get("CORS_ORIGINS").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

supabase_repository = SupabaseRepository()

chroma_repository = ChromaRepository()

embedding_provider = LocalSentenceTransformerProvider(
    model_name=settings.embedding_model or "intfloat/multilingual-e5-large",
    batch_size=settings.embedding_batch_size,
    device=settings.embedding_device,
    fp16=settings.embedding_fp16,
)

dataset_service = DatasetService(
    supabase_repository=supabase_repository,
    chroma_repository=chroma_repository,
    embedding_provider=embedding_provider,
)

class SearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
    )

    n_results: int = Field(
        default=10,
        ge=1,
        le=100,
    )

    dataset_id: str | None = None

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


@app.post("/uploadDataset", status_code=202)
async def start_upload(
    background_tasks: BackgroundTasks,
    archivo: UploadFile = File(...),

    columnaTexto: str = Form(...),

    columnaClase: str | None = Form(
        default=None
    ),

    columnaFecha: str | None = Form(
        default=None
    ),

    columnaUsuario: str | None = Form(
        default=None
    ),

    formatoOriginal: str | None = Form(
        default=None
    ),

    formatoDetectado: str | None = Form(
        default=None
    ),

    formatoFecha: str | None = Form(
        default=None
    ),

    esChat: bool = Form(
        default=False
    ),

    context: Annotated[
        RequestContext,
        Depends(get_request_context),
    ] = None,

    x_user_id: str = Header(...),
    x_user_token: str = Header(...),
):

    if not archivo.filename:

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El archivo no tiene nombre.",
        )

    extension = os.path.splitext(
        archivo.filename
    )[1].lower()

    allowed_extensions = {
        ".csv",
        ".xlsx",
    }

    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=(
                status.HTTP_422_UNPROCESSABLE_ENTITY
            ),
            detail=(
                f"Formato '{extension}' "
                f"no soportado. "
                f"Formatos permitidos: "
                f"{', '.join(sorted(allowed_extensions))}"
            ),
        )

    max_size = (
        settings.max_upload_size_mb
        * 1024
        * 1024
    )


    upload_tmp: Path | None = None
    tamaño = 0
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=extension
        ) as tmp:
            upload_tmp = Path(tmp.name)

            while True:
                pedazo = await archivo.read(1024 * 1024)

                if not pedazo:
                    break

                tamaño += len(pedazo)

                if tamaño > max_size:
                    raise HTTPException(
                        status_code=(
                            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
                        ),
                        detail=(
                            f"El archivo supera el límite "
                            f"de {settings.max_upload_size_mb} MB."
                        ),
                    )

                tmp.write(pedazo)

        if tamaño == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="El archivo está vacío.",
            )

        content = upload_tmp.read_bytes()
    finally:
        if upload_tmp is not None:
            upload_tmp.unlink(missing_ok=True)

    try:
        supabase = supabase_repository.get_supabase_client(x_user_token)
        job = jobs.create_job(
            supabase, service="dataset", job_type="upload",
            user_id=x_user_id, dataset_name=archivo.filename,
            params={
                "filename": archivo.filename,
                "columna_texto": columnaTexto,
                "columna_clase": columnaClase,
                "columna_fecha": columnaFecha,
                "columna_autor": columnaUsuario,
                "formato_original": formatoOriginal,
                "formato_detectado": formatoDetectado,
                "formato_fecha": formatoFecha,
                "es_chat": esChat,
            },
        )
    except Exception as exc:
        logger.exception(
            f"[dataset-service] fallo creando job upload | "
            f"method=POST user_id={x_user_id} causa={type(exc).__name__}: {exc}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno del servidor.",
        ) from exc

    background_tasks.add_task(
        _run_upload_job, job["job_id"], content,
        {
            "filename": archivo.filename,
            "columna_texto": columnaTexto,
            "columna_clase": columnaClase,
            "columna_fecha": columnaFecha,
            "columna_autor": columnaUsuario,
            "formato_original": formatoOriginal,
            "formato_detectado": formatoDetectado,
            "formato_fecha": formatoFecha,
            "es_chat": esChat,
        },
        x_user_id, x_user_token,
    )
    return {"job_id": job["job_id"], "status": "queued", "filename": archivo.filename}


async def _run_upload_job(job_id: str, content: bytes, params: dict, user_id: str, user_token: str):
    """Procesa la carga en background (parse + embeddings + Chroma)."""
    supabase = supabase_repository.get_supabase_client(user_token)
    jobs.mark_running(supabase, job_id)
    try:
        formato_fecha = params.get("formato_fecha")
        if formato_fecha and formato_fecha not in ("DMY", "MDY", "YMD"):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"formatoFecha debe ser DMY, MDY o YMD, recibido: {formato_fecha}",
            )

        dataset = await dataset_service.create_dataset(
            content=content,
            filename=params["filename"],
            columna_texto=params["columna_texto"],
            columna_clase=params.get("columna_clase"),
            columna_fecha=params.get("columna_fecha"),
            columna_autor=params.get("columna_autor"),
            formato_original=params.get("formato_original"),
            formato_detectado=params.get("formato_detectado"),
            formato_fecha=formato_fecha,
            es_chat=params.get("es_chat", False),
            user_id=user_id,
            user_token=user_token,
        )

        result = {
            "status": "success",
            "dataset_id": dataset.dataset_id,
            "filename": dataset.filename,
            "total_registros_guardados": dataset.total_documents,
            "created_at": dataset.created_at,
        }
    except HTTPException as e:
        logger.warning(f"[dataset-service] job {job_id} fallo | code={e.status_code} detail={e.detail}")
        jobs.mark_failed(supabase, job_id, e.status_code, str(e.detail))
        return
    except ValueError as exc:
        logger.exception(
            f"[dataset-service] job {job_id} fallo validando dataset | causa={type(exc).__name__}"
        )
        jobs.mark_failed(supabase, job_id, 422, str(exc))
        return
    except Exception as exc:
        logger.exception(
            f"[dataset-service] job {job_id} fallo no controlado | causa={type(exc).__name__}: {exc}"
        )
        jobs.mark_failed(supabase, job_id, 500, "Error interno procesando el dataset.")
        return
    jobs.mark_succeeded(supabase, job_id, jsonable_encoder(result))


@app.get("/jobs/{job_id}")
async def get_job_status(
    job_id: str,
    x_user_id: str = Header(...),
    x_user_token: str = Header(...),
):
    supabase = supabase_repository.get_supabase_client(x_user_token)
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
    supabase = supabase_repository.get_supabase_client(x_user_token)
    return {"jobs": jobs.list_jobs(supabase, x_user_id, dataset_name, job_type, active_only=True)}


@app.get("/getDatasets")
async def getDataset(
    x_user_id: str = Header(...),
    x_user_token: str = Header(...),
):
    try:
        datasets = (
            await dataset_service.get_datasets(
                user_id=x_user_id,
                user_token=x_user_token
            )
        )
        return {
            "datasets": datasets,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(
            "[dataset-service] fallo en GET /getDatasets | "
            f"method=GET user_id={x_user_id} causa={type(e).__name__}: {e}"
        )
        raise HTTPException(status_code=500, detail="Error interno obteniendo datasets.")

@app.delete("/deleteDataset")
async def deleteDataset(
    background_tasks: BackgroundTasks,
    dataset_id: str,
    x_user_id: str = Header(...),
    x_user_token: str = Header(...),
):
    try:
        await dataset_service.delete_dataset(
            dataset_id=dataset_id,
            user_id=x_user_id,
            user_token=x_user_token
        )
        background_tasks.add_task(
            dataset_service.delete_dataset_vectors,
            dataset_id=dataset_id,
        )
        return {
            "status": "success",
            "message": f"Dataset {dataset_id} eliminado correctamente."
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(
            "[dataset-service] fallo en DELETE /deleteDataset | "
            f"method=DELETE user_id={x_user_id} dataset_id={dataset_id} "
            f"causa={type(e).__name__}: {e}"
        )
        raise HTTPException(status_code=500, detail="Error interno eliminando dataset.")

@app.delete("/deleteDatasetsByUser")
async def deleteDatasetsByUser(
    background_tasks: BackgroundTasks,
    x_user_id: str = Header(...),
    x_user_token: str = Header(...),
):
    try:
        await dataset_service.delete_datasets_by_user(
            user_id=x_user_id,
            user_token=x_user_token
        )
        background_tasks.add_task(
            dataset_service.delete_user_vectors,
            user_id=x_user_id,
        )
        return {
            "status": "success",
            "message": f"Todos los datasets del usuario {x_user_id} eliminados correctamente."
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(
            "[dataset-service] fallo en DELETE /deleteDatasetsByUser | "
            f"method=DELETE user_id={x_user_id} causa={type(e).__name__}: {e}"
        )
        raise HTTPException(status_code=500, detail="Error interno eliminando datasets.")
