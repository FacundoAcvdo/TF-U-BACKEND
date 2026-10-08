import os
import tempfile
from pathlib import Path
from typing import Annotated

from contextlib import asynccontextmanager
import logging
import torch

from fastapi import (
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

settings = get_settings()

logger = logging.getLogger("uvicorn.info")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Verificación de GPU al iniciar ---
    logger.info("=" * 55)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        vram_reserved_gb = torch.cuda.memory_reserved(0) / (1024**3)
        vram_free_gb = vram_total_gb - vram_reserved_gb

        logger.info("🚀 [GPU DETECTADA] Soporte CUDA activo")
        logger.info(f"   Placa:        {gpu_name}")
        logger.info(f"   VRAM Total:   {vram_total_gb:.2f} GB")
        logger.info(f"   VRAM Libre:   {vram_free_gb:.2f} GB")
    else:
        logger.warning("⚠️  [MODO CPU] CUDA no disponible. Corriendo en procesador.")
    logger.info("=" * 55)

    yield

    # --- Limpieza al apagar el contenedor ---
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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

@app.post("/uploadDataset")
async def create_dataset(
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

    # Spool a disco por chunks de 1 MB: RAM acotada y aborto temprano
    # con 413 en cuanto se supera el límite (sin leer todo el archivo).
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

        if formatoFecha and formatoFecha not in ("DMY", "MDY", "YMD"):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"formatoFecha debe ser DMY, MDY o YMD, recibido: {formatoFecha}",
            )

        dataset = (
            await dataset_service.create_dataset(
                content=content,
                filename=archivo.filename,
                columna_texto=columnaTexto,
                columna_clase=columnaClase,
                columna_fecha=columnaFecha,
                columna_autor=columnaUsuario,
                formato_original=formatoOriginal,
                formato_detectado=formatoDetectado,
                formato_fecha=formatoFecha,
                es_chat=esChat,
                user_id=x_user_id,
                user_token=x_user_token
            )
        )

        return {
            "status": "success",
            "dataset_id": dataset.dataset_id,
            "filename": dataset.filename,
            "total_registros_guardados": (
                dataset.total_documents
            ),
            "created_at": dataset.created_at,
        }

    except ValueError as exc:

        raise HTTPException(
            status_code=(
                status.HTTP_422_UNPROCESSABLE_ENTITY
            ),
            detail=str(exc),
        ) from exc

    except Exception as exc:

        print(
            f"Error creando dataset: {exc}"
        )

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Error interno procesando "
                "el dataset."
            ),
        ) from exc

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

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))