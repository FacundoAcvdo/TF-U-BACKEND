from typing import Annotated

from fastapi import (
    FastAPI,
    Header,
    HTTPException,
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from config import get_settings

from dataset_storage_service import (
    DatasetStorageService,
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

dataset_storage_service = (
    DatasetStorageService(
        settings.dataset_storage_path
    )
)

download_service = DownloadService(
    storage_path=settings.cleaning_download_path
)

supabase_service = SupabaseService()

cleaning_service = CleaningService(
    dataset_storage_service=dataset_storage_service,
    download_service=download_service,
    supabase_service=supabase_service,
)

@app.post("/cleanDataset")
async def clean_dataset(
    request: CleaningRequest,
    x_user_id: str = Header(...),
    x_user_token: str = Header(...),
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

        print(
            f"[CLEANING VALUE ERROR] "
            f"{type(exc).__name__}: {exc}"
        )

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

        print(
            f"Error ejecutando cleaning: {exc}"
        )

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

        print(
            f"Error descargando archivo: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Error descargando "
                "el archivo."
            ),
        ) from exc