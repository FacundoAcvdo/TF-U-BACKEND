import os
from typing import Annotated

from fastapi import (
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
from dataset_storage import (
    LocalJsonDatasetStorage,
)
from dependencies import (
    RequestContext,
    get_request_context,
)
from embeddings import (
    NoEmbeddingProvider,
)
from services import DatasetService
from supabase_repository import (
    SupabaseRepository,
)

from fastapi.middleware.cors import CORSMiddleware

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

dataset_storage = LocalJsonDatasetStorage(
    base_path=settings.dataset_storage_path
)

supabase_repository = SupabaseRepository()

chroma_repository = ChromaRepository()

embedding_provider = NoEmbeddingProvider()

dataset_service = DatasetService(
    dataset_storage=dataset_storage,
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

    formatoOriginal: str | None = Form(
        default=None
    ),

    formatoDetectado: str | None = Form(
        default=None
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

    content = await archivo.read()

    max_size = (
        settings.max_upload_size_mb
        * 1024
        * 1024
    )

    if len(content) > max_size:

        raise HTTPException(
            status_code=(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
            ),
            detail=(
                f"El archivo supera el límite "
                f"de {settings.max_upload_size_mb} MB."
            ),
        )

    if not content:

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El archivo está vacío.",
        )

    try:

        dataset = (
            await dataset_service.create_dataset(
                content=content,
                filename=archivo.filename,
                columna_texto=columnaTexto,
                columna_clase=columnaClase,
                columna_fecha=columnaFecha,
                formato_original=formatoOriginal,
                formato_detectado=formatoDetectado,
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